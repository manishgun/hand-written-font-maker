import { useEffect, useState, useRef, useCallback } from "react";
import opencv from "@techstark/opencv-js";
import opentype from "opentype.js";
import { DOMParser } from "xmldom";
import svgpath from "svgpath";
import confetti from "canvas-confetti";
import { characters } from "./constants";

const TYPE_SEQUENCE = characters;

type Step = "idle" | "grayscale" | "denoise" | "contrast" | "threshold" | "edges" | "corners" | "warped" | "blocks" | "extracting" | "adjusting" | "done";

interface Glyph {
  type: string;
  svg: string;
  originalImg: string;
}

interface AdjustableGlyph {
  character: string;
  originalImg: string;
  xOffset: number;
  yOffset: number;
  scale: number;
  rotation: number;
}
interface ProcessingSnapshot {
  name: string;
  image: string;
}

// ─── Winding-order utilities ───────────────────────────────────────────────

/** Shoelace signed area. Positive → CCW (Y-up). Negative → CW. */
function signedArea(pts: { x: number; y: number }[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i],
      q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

function flattenSegments(segs: any[]): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  let cx = 0,
    cy = 0;
  for (const s of segs) {
    switch (s[0]) {
      case "M":
        cx = s[1];
        cy = s[2];
        pts.push({ x: cx, y: cy });
        break;
      case "L":
        cx = s[1];
        cy = s[2];
        pts.push({ x: cx, y: cy });
        break;
      case "H":
        cx = s[1];
        pts.push({ x: cx, y: cy });
        break;
      case "V":
        cy = s[1];
        pts.push({ x: cx, y: cy });
        break;
      case "C":
        pts.push({ x: (cx + s[5]) / 2, y: (cy + s[6]) / 2 });
        cx = s[5];
        cy = s[6];
        pts.push({ x: cx, y: cy });
        break;
      case "Q":
        cx = s[3];
        cy = s[4];
        pts.push({ x: cx, y: cy });
        break;
    }
  }
  return pts;
}

/**
 * Reverse a contour while PRESERVING Bezier curves.
 *
 * Why curves must be preserved:
 *   After the Y-flip transform, winding is inverted, so we call this function
 *   to correct it. The previous implementation sampled only endpoint 'anchors'
 *   and emitted LineTo for everything — turning smooth ovals into polygons and
 *   causing the "non-continuous" jagged look in O, P, Q, etc.
 *
 * Bezier reversal rules (mathematically correct):
 *   Cubic  "C c1x c1y c2x c2y ex ey" reversed from [start] →
 *          "C c2x c2y c1x c1y start.x start.y"  (swap control points)
 *   Quad   "Q cx cy ex ey" reversed from [start] →
 *          "Q cx cy start.x start.y"             (control point unchanged)
 *   Linear "L/H/V …" reversed from [start] →
 *          "L start.x start.y"
 */
function reverseContour(segs: any[]): any[] {
  if (segs.length <= 1) return segs;

  type PS = { type: string; start: [number, number]; args: number[] };
  const parsed: PS[] = [];
  let cx = 0,
    cy = 0;

  for (const s of segs) {
    const type = s[0] as string;
    const args = s.slice(1) as number[];
    parsed.push({ type, start: [cx, cy], args });
    // Track current pen position
    if (type === "M" || type === "L") {
      cx = args[0];
      cy = args[1];
    } else if (type === "H") {
      cx = args[0];
    } else if (type === "V") {
      cy = args[0];
    } else if (type === "C") {
      cx = args[4];
      cy = args[5];
    } else if (type === "Q") {
      cx = args[2];
      cy = args[3];
    }
    // Z: in our tracker cx/cy stays as the last drawing endpoint — correct
  }

  // The new M starts at wherever the last drawing command ended (cx, cy)
  const out: any[] = [["M", cx, cy]];

  for (let i = parsed.length - 1; i >= 0; i--) {
    const { type, start, args } = parsed[i];
    if (type === "M" || type === "Z") continue;

    if (type === "L" || type === "H" || type === "V") {
      // Reversed line: just draw back to the segment's start point
      out.push(["L", start[0], start[1]]);
    } else if (type === "C") {
      // Cubic: endpoint becomes start, swap C1 ↔ C2
      // Original: C args[0] args[1]  args[2] args[3]  args[4] args[5]
      //              c1x     c1y      c2x     c2y      ex      ey
      // Reversed: C args[2] args[3]  args[0] args[1]  start.x start.y
      out.push(["C", args[2], args[3], args[0], args[1], start[0], start[1]]);
    } else if (type === "Q") {
      // Quadratic: control point is invariant under reversal, endpoint → start
      // Original: Q args[0] args[1]  args[2] args[3]
      //              cx       cy      ex      ey
      // Reversed: Q args[0] args[1]  start.x start.y
      out.push(["Q", args[0], args[1], start[0], start[1]]);
    }
  }

  out.push(["Z"]);
  return out;
}

function splitSubPaths(d: string): string[] {
  const tokens = d.trim().split(/(?=[Mm])/);
  const parts = tokens.map((t) => t.trim()).filter(Boolean);
  return parts.length ? parts : [d];
}

function emitToPath(path: opentype.Path, segs: any[]): void {
  let cx = 0,
    cy = 0;
  for (const s of segs) {
    switch (s[0]) {
      case "M":
        path.moveTo(s[1], s[2]);
        cx = s[1];
        cy = s[2];
        break;
      case "L":
        path.lineTo(s[1], s[2]);
        cx = s[1];
        cy = s[2];
        break;
      case "C":
        path.curveTo(s[1], s[2], s[3], s[4], s[5], s[6]);
        cx = s[5];
        cy = s[6];
        break;
      case "Q":
        path.quadTo(s[1], s[2], s[3], s[4]);
        cx = s[3];
        cy = s[4];
        break;
      case "Z":
        path.close();
        break;
      case "H":
        path.lineTo(s[1], cy);
        cx = s[1];
        break;
      case "V":
        path.lineTo(cx, s[1]);
        cy = s[1];
        break;
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// OpenCV-based glyph tracer
// Replaces Potrace: uses cv.findContours (RETR_CCOMP) so outer/hole hierarchy
// is known structurally — no winding math guesswork.
// Contours are smoothed with approxPolyDP then promoted to cubic beziers via
// Catmull-Rom → cubic conversion for smooth curves.
// ──────────────────────────────────────────────────────────────────────────────

function catmullToCubic(
  p0: { x: number; y: number },
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  p3: { x: number; y: number },
): [number, number, number, number, number, number] {
  // Convert a Catmull-Rom segment (p1→p2) to a cubic bezier
  const alpha = 1 / 6;
  return [p1.x + alpha * (p2.x - p0.x), p1.y + alpha * (p2.y - p0.y), p2.x - alpha * (p3.x - p1.x), p2.y - alpha * (p3.y - p1.y), p2.x, p2.y];
}

function contourToSVGPath(pts: { x: number; y: number }[], close = true): string {
  if (pts.length < 2) return "";
  const n = pts.length;
  let d = `M ${pts[0].x} ${pts[0].y} `;
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n];
    const p1 = pts[i % n];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];
    const [c1x, c1y, c2x, c2y, ex, ey] = catmullToCubic(p0, p1, p2, p3);
    d += `C ${c1x.toFixed(2)} ${c1y.toFixed(2)} ${c2x.toFixed(2)} ${c2y.toFixed(2)} ${ex.toFixed(2)} ${ey.toFixed(2)} `;
  }
  if (close) d += "Z";
  return d.trim();
}

function traceGlyphWithOpenCV(cv: typeof opencv, croppedCanvas: HTMLCanvasElement, boldness: number, fidelity: number): string {
  const w = croppedCanvas.width;
  const h = croppedCanvas.height;

  // Read image into OpenCV
  const src = cv.imread(croppedCanvas);

  // Convert to grayscale
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  // Threshold: Otsu finds the optimal cut between ink and background
  const binary = new cv.Mat();
  cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

  // Light denoise: remove specks smaller than ~3x3
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  const cleaned = new cv.Mat();
  cv.morphologyEx(binary, cleaned, cv.MORPH_OPEN, kernel);

  // Apply Boldness (Dilate to thicken, Erode to thin)
  // Multiplied by 2 for more visible impact on high-res scans
  if (boldness > 0) {
    const kSize = boldness * 3 + 1;
    const dKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kSize, kSize));
    cv.dilate(cleaned, cleaned, dKernel);
    dKernel.delete();
  } else if (boldness < 0) {
    const kSize = Math.abs(boldness) * 3 + 1;
    const eKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kSize, kSize));
    cv.erode(cleaned, cleaned, eKernel);
    eKernel.delete();
  }

  // Find contours with full 2-level hierarchy (RETR_CCOMP)
  // Level 0 = outer fill contours, Level 1 = hole contours inside them
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(cleaned, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE);

  const minArea = w * h * 0.0003; // ignore specks < 0.03% of cell area

  // Build paths: walk the hierarchy
  // hierarchy[i] = [next, prev, firstChild, parent]
  // parent === -1  → outer contour (level 0)
  // parent !== -1  → hole contour  (level 1)
  const pathParts: string[] = [];

  for (let i = 0; i < (contours.size() as unknown as number); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt);
    if (area < minArea) {
      cnt.delete();
      continue;
    }

    const hData = hierarchy.intPtr(0, i);
    const isHole = hData[3] !== -1; // has a parent → hole

    // Smooth the contour
    const approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, fidelity, true);

    const pts: { x: number; y: number }[] = [];
    for (let j = 0; j < approx.rows; j++) {
      pts.push({ x: approx.data32S[j * 2], y: approx.data32S[j * 2 + 1] });
    }
    approx.delete();
    cnt.delete();

    if (pts.length < 3) continue;

    // Enforce winding: OpenType outer=CW(Y-up)=CCW(Y-down), hole=CCW(Y-up)=CW(Y-down)
    // In image space (Y-down), outer must be CCW, hole must be CW
    let area2 = 0;
    for (let j = 0; j < pts.length; j++) {
      const a = pts[j],
        b = pts[(j + 1) % pts.length];
      area2 += a.x * b.y - b.x * a.y;
    }
    const isCCW = area2 > 0; // positive shoelace = CCW in Y-down
    if (!isHole && !isCCW) pts.reverse(); // outer must be CCW in Y-down
    if (isHole && isCCW) pts.reverse(); // hole must be CW in Y-down

    pathParts.push(contourToSVGPath(pts, true));
  }

  // Cleanup
  src.delete();
  gray.delete();
  binary.delete();
  kernel.delete();
  cleaned.delete();
  contours.delete();
  hierarchy.delete();

  if (pathParts.length === 0) return "";

  const allD = pathParts.join(" ");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><path d="${allD}" fill="black" fill-rule="nonzero"/></svg>`;
}

// ── GlyphCard ─────────────────────────────────────────────────────────────────
// Drag is handled by tracking live position in a ref (livePos) and updating
// React state ONLY on mouseup — avoiding the removeChild crash that happens
// when React tries to reconcile nodes we mutated directly via imgRef.
// The image transform reads from React state normally; during drag we use
// requestAnimationFrame to flush the committed state fast enough to feel instant.
interface GlyphCardProps {
  ag: AdjustableGlyph;
  i: number;
  onUpdate: (i: number, updates: Partial<AdjustableGlyph>) => void;
}

function GlyphCard({ ag, i, onUpdate }: GlyphCardProps) {
  // livePos: tracks drag position without triggering re-renders
  const livePos = useRef({ x: ag.xOffset, y: ag.yOffset });
  const isDragging = useRef(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  // Keep livePos in sync when parent updates (slider, reset)
  useEffect(() => {
    livePos.current = { x: ag.xOffset, y: ag.yOffset };
    // Sync overlay text when React state changes (slider / reset) — safe because no JSX children
    if (overlayRef.current) {
      const x = ag.xOffset,
        y = ag.yOffset;
      overlayRef.current.textContent = `${x >= 0 ? "+" : ""}${Math.round(x)}, ${y >= 0 ? "+" : ""}${Math.round(y)}`;
    }
  }, [ag.xOffset, ag.yOffset]);

  const startDrag = useCallback(
    (clientX: number, clientY: number, rectW: number, rectH: number) => {
      isDragging.current = true;
      const startXOff = ag.xOffset;
      const startYOff = ag.yOffset;

      const updateOverlay = (nx: number, ny: number) => {
        if (overlayRef.current) {
          overlayRef.current.textContent = `${nx >= 0 ? "+" : ""}${Math.round(nx)}, ${ny >= 0 ? "+" : ""}${Math.round(ny)}`;
          overlayRef.current.style.opacity = "1";
        }
      };

      const onMove = (cx: number, cy: number) => {
        const dx = ((cx - clientX) / rectW) * 100;
        const dy = ((cy - clientY) / rectH) * 100;
        const nx = Math.max(-50, Math.min(50, startXOff + dx));
        const ny = Math.max(-50, Math.min(50, startYOff + dy));
        livePos.current = { x: nx, y: ny };
        // Batch overlay update in rAF — smooth but never touches React-owned nodes
        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(() => updateOverlay(nx, ny));
        // Commit to React via onUpdate on every move — React batches these efficiently
        onUpdate(i, { xOffset: nx, yOffset: ny });
      };

      const commit = () => {
        isDragging.current = false;
        cancelAnimationFrame(rafRef.current);
        window.removeEventListener("mousemove", mm);
        window.removeEventListener("mouseup", mu);
        window.removeEventListener("touchmove", tm);
        window.removeEventListener("touchend", te);
      };

      const mm = (e: MouseEvent) => onMove(e.clientX, e.clientY);
      const mu = () => commit();
      const tm = (e: TouchEvent) => {
        e.preventDefault();
        onMove(e.touches[0].clientX, e.touches[0].clientY);
      };
      const te = () => commit();
      window.addEventListener("mousemove", mm);
      window.addEventListener("mouseup", mu);
      window.addEventListener("touchmove", tm, { passive: false });
      window.addEventListener("touchend", te);
    },
    [ag.xOffset, ag.yOffset, i, onUpdate],
  );

  const onMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      const r = e.currentTarget.getBoundingClientRect();
      startDrag(e.clientX, e.clientY, r.width, r.height);
    },
    [startDrag],
  );

  const onTouchStart = useCallback(
    (e: React.TouchEvent<HTMLDivElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      startDrag(e.touches[0].clientX, e.touches[0].clientY, r.width, r.height);
    },
    [startDrag],
  );

  const isDirty = ag.xOffset !== 0 || ag.yOffset !== 0 || ag.scale !== 1 || ag.rotation !== 0;

  return (
    <div className="bg-white border border-slate-100 rounded-2xl p-3 shadow-sm flex flex-col gap-2.5 hover:border-sky-200 hover:shadow-lg transition-all duration-200 group">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl font-black text-slate-900 leading-none w-6">{ag.character}</span>
          <span className="text-[8px] font-semibold text-slate-300">#{i + 1}</span>
        </div>
        {isDirty && (
          <button
            onClick={() => onUpdate(i, { xOffset: 0, yOffset: 0, scale: 1, rotation: 0 })}
            className="text-[8px] font-bold text-rose-400 hover:text-rose-600 px-1.5 py-0.5 rounded-md bg-rose-50 hover:bg-rose-100 transition-colors">
            ↺ Reset
          </button>
        )}
      </div>

      {/* Drag zone — React owns the img node, guides are z-10 above it */}
      <div
        ref={containerRef}
        className="aspect-square bg-gradient-to-br from-slate-50 to-slate-100/50 rounded-xl overflow-hidden flex items-center justify-center relative cursor-grab active:cursor-grabbing select-none border border-slate-100 group-hover:border-sky-200 transition-colors"
        onMouseDown={onMouseDown}
        onTouchStart={onTouchStart}>
        {/* Image — React-controlled transform, no direct DOM mutation */}
        <img
          src={ag.originalImg}
          className="relative z-[1] max-w-[72%] max-h-[72%] object-contain pointer-events-none select-none"
          style={{
            transform: `translate(${ag.xOffset}%, ${ag.yOffset}%) scale(${ag.scale}) rotate(${ag.rotation}deg)`,
            willChange: "transform",
            transition: isDragging.current ? "none" : "transform 0.08s ease-out",
          }}
          draggable={false}
          alt={ag.character}
        />

        {/* Typographic Guides (Notebook Style) — Overlaid on top of image for precision */}
        <div className="absolute inset-0 z-10 flex flex-col justify-center pointer-events-none opacity-50">
          {/* Ascender line */}
          <div className="w-full h-px border-t border-dashed border-slate-400 -translate-y-12" />
          {/* X-Height line */}
          <div className="w-full h-px border-t border-dotted border-sky-400 -translate-y-4" />
          {/* Baseline (Main) */}
          <div className="w-full h-[2px] bg-indigo-400 translate-y-8 shadow-sm" />
          {/* Descender line */}
          <div className="w-full h-px border-t border-dashed border-slate-300 translate-y-16" />
        </div>

        {/* Crosshair guides — z-10 so they sit above the image */}
        <div className="absolute z-10 top-1/2 left-0 w-full h-px bg-sky-400/25 pointer-events-none" />
        <div className="absolute z-10 left-1/2 top-0 w-px h-full bg-sky-400/25 pointer-events-none" />

        {/* Live XY overlay — intentionally NO JSX children so React never owns a text node here.
             We write textContent directly. React only controls style/className on this div. */}
        <div
          ref={overlayRef}
          className="absolute z-10 top-1.5 left-1.5 text-[7px] font-mono font-bold text-sky-500 bg-white/90 px-1.5 py-0.5 rounded-full border border-sky-100 shadow-sm pointer-events-none leading-none"
          style={{ opacity: ag.xOffset !== 0 || ag.yOffset !== 0 ? 1 : 0, transition: "opacity 0.2s" }}
        />

        {/* Drag hint */}
        <div className="absolute z-10 bottom-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <span className="text-[7px] font-bold text-slate-400 bg-white/80 px-1.5 py-0.5 rounded-full border border-slate-100">⠿ drag</span>
        </div>
      </div>

      {/* Sliders */}
      <div className="flex flex-col gap-1.5 px-0.5">
        <div className="flex items-center gap-1.5">
          <span className="text-[8px] font-black text-slate-400 uppercase tracking-wider w-5 shrink-0">Sc</span>
          <input
            type="range"
            min="0.4"
            max="2.5"
            step="0.05"
            value={ag.scale}
            onChange={(e) => onUpdate(i, { scale: parseFloat(e.target.value) })}
            className="flex-1 h-[3px] rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
          />
          <span className="text-[8px] font-bold text-sky-500 w-6 text-right tabular-nums">{ag.scale.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[8px] font-black text-slate-400 uppercase tracking-wider w-5 shrink-0">Ro</span>
          <input
            type="range"
            min="-45"
            max="45"
            step="1"
            value={ag.rotation}
            onChange={(e) => onUpdate(i, { rotation: parseFloat(e.target.value) })}
            className="flex-1 h-[3px] rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
          />
          <span className="text-[8px] font-bold text-sky-500 w-6 text-right tabular-nums">{ag.rotation}°</span>
        </div>
        <div className="grid grid-cols-2 gap-2 pt-1 mt-0.5 border-t border-slate-100">
          <div className="flex flex-col gap-0.5">
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-black text-slate-400 uppercase">X</span>
              <span className="text-[8px] font-bold text-sky-500 tabular-nums">{Math.round(ag.xOffset)}</span>
            </div>
            <input
              type="range"
              min="-50"
              max="50"
              step="1"
              value={ag.xOffset}
              onChange={(e) => onUpdate(i, { xOffset: parseFloat(e.target.value) })}
              className="w-full h-[3px] rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
            />
          </div>
          <div className="flex flex-col gap-0.5">
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-black text-slate-400 uppercase">Y</span>
              <span className="text-[8px] font-bold text-sky-500 tabular-nums">{Math.round(ag.yOffset)}</span>
            </div>
            <input
              type="range"
              min="-50"
              max="50"
              step="1"
              value={ag.yOffset}
              onChange={(e) => onUpdate(i, { yOffset: parseFloat(e.target.value) })}
              className="w-full h-[3px] rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [cv, setCV] = useState<typeof opencv | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [currentStep, setCurrentStep] = useState<Step>("idle");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingFont, setIsGeneratingFont] = useState(false);
  const [glyphs, setGlyphs] = useState<Glyph[]>([]);
  const [adjustableGlyphs, setAdjustableGlyphs] = useState<AdjustableGlyph[]>([]);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [history, setHistory] = useState<ProcessingSnapshot[]>([]);
  const [appliedFontName, setAppliedFontName] = useState<string>("inherit");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inventoryRef = useRef<HTMLDivElement>(null);
  const [previewFontSize, setPreviewFontSize] = useState(32);
  const [globalScale, setGlobalScale] = useState(2.45);
  const [globalYOffset, setGlobalYOffset] = useState(0);
  const [showComparison, setShowComparison] = useState(true);
  const [previewText, setPreviewText] = useState("The quick brown fox jumps over the lazy dog.");
  const [familyName, setFamilyName] = useState("");
  const [styleName, setStyleName] = useState("Handwriting");
  const [designer, setDesigner] = useState("");
  const [version, setVersion] = useState("1.000");
  const [description, setDescription] = useState("Generated using Handwritten Font Maker");
  const [letterSpacing, setLetterSpacing] = useState(25);
  const [wordSpacing, setWordSpacing] = useState(300);
  const [slant, setSlant] = useState(0);
  const [boldness, setBoldness] = useState(0);
  const [fidelity, setFidelity] = useState(0.5);
  const [generationProgress, setGenerationProgress] = useState(0);

  useEffect(() => {
    opencv.onRuntimeInitialized = () => setCV(opencv);
  }, []);

  useEffect(() => {
    if (isProcessing && glyphs.length > 0) window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }, [glyphs.length, isProcessing]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const url = ev.target?.result as string;
      setProcessedImage(url);
      setHistory([{ name: "Source", image: url }]);
    };
    reader.readAsDataURL(file);
    setCurrentStep("idle");
    setGlyphs([]);
    setAdjustableGlyphs([]);
    setAppliedFontName("inherit");
  };

  const resetParams = () => {
    setGlobalScale(2.45);
    setGlobalYOffset(0);
    setLetterSpacing(25);
    setWordSpacing(300);
    setSlant(0);
    setBoldness(0);
    setFidelity(0.5);
  };

  // Expensive Auto-regeneration (Re-traces via OpenCV)
  useEffect(() => {
    if (isDone || isAdjusting) {
      const timer = setTimeout(() => {
        if (!isGeneratingFont && !isProcessing) {
          generateFont(true); // Silent update
        }
      }, 750);
      return () => clearTimeout(timer);
    }
  }, [boldness, fidelity]);

  // Cheap Auto-update (Re-buffers vs Metadata/Metrics)
  useEffect(() => {
    if (isDone && glyphs.length > 0) {
      const timer = setTimeout(() => {
        if (!isGeneratingFont) {
          applyGeneratedFont(glyphs, true);
        }
      }, 80);
      return () => clearTimeout(timer);
    }
  }, [globalScale, globalYOffset, letterSpacing, wordSpacing, slant, familyName, styleName, designer, version, description]);

  const reset = () => {
    setImageFile(null);
    setGlyphs([]);
    setAdjustableGlyphs([]);
    setHistory([]);
    setCurrentStep("idle");
    setAppliedFontName("inherit");
    setProcessedImage(null);
  };

  const processImage = async () => {
    if (!cv || !imageFile || !canvasRef.current) return;
    setIsProcessing(true);
    setGlyphs([]);
    setAppliedFontName("inherit");
    setHistory((prev) => [prev[0]]);

    const img = new Image();
    img.src = history[0].image;
    await new Promise((r) => (img.onload = r));

    const canvas = canvasRef.current;
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d")!.drawImage(img, 0, 0);
    const src = cv.imread(canvas);

    const snap = async (step: Step, mat: any, name: string, delay = 700, decorate?: (ctx: CanvasRenderingContext2D, c: HTMLCanvasElement) => void) => {
      setCurrentStep(step);
      cv.imshow(canvas, mat);
      if (decorate) decorate(canvas.getContext("2d")!, canvas);
      const url = canvas.toDataURL();
      setProcessedImage(url);
      setHistory((p) => [...p, { name, image: url }]);
      await new Promise((r) => setTimeout(r, delay));
    };

    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    await snap("grayscale", gray, "Grayscale");

    // Visual Stage: Denoise
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    await snap("denoise", blurred, "Antialiasing");

    const hdr = new cv.Mat();

    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    clahe.apply(blurred, hdr);

    // Visual Stage: Contrast
    const contrast = new cv.Mat();
    hdr.convertTo(contrast, -1, 1.8, 1.4);
    await snap("contrast", contrast, "Luma Correction");

    const thresh = new cv.Mat();
    // Use an adaptive block size based on image resolution
    const adaptiveBlockSize = Math.floor(Math.min(src.cols, src.rows) / 100) * 2 + 1;
    cv.adaptiveThreshold(gray, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, Math.max(11, adaptiveBlockSize), 10);
    await snap("threshold", thresh, "Threshold");

    // ─── Marker Detection 3.0 ──────────────────────────────────────────
    // Add 2px padding to handle markers touching the edge
    const padded = new cv.Mat();
    cv.copyMakeBorder(thresh, padded, 2, 2, 2, 2, cv.BORDER_CONSTANT, new cv.Scalar(0));

    const pContours = new cv.MatVector();
    const pHier = new cv.Mat();
    cv.findContours(padded, pContours, pHier, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    const candidates: { r: opencv.Rect; score: number; index: number }[] = [];
    const minMSize = Math.min(src.cols, src.rows) * 0.005; // Loosened from 0.01
    const maxMSize = Math.min(src.cols, src.rows) * 0.3;

    for (let i = 0; i < pContours.size(); i++) {
      const cnt = pContours.get(i);
      const r = cv.boundingRect(cnt);
      // Correct for padding
      r.x -= 2;
      r.y -= 2;

      const area = cv.contourArea(cnt);

      if (r.width < minMSize || r.height < minMSize || r.width > maxMSize || r.height > maxMSize) continue;

      const aspectRatio = r.width / r.height;
      if (aspectRatio < 0.6 || aspectRatio > 1.6) continue;

      const extent = area / (r.width * r.height);
      if (extent < 0.3) continue;

      // Level 2 Hierarchy (at least one child) is worth scoring
      const hData = pHier.intPtr(0, i);
      const childIdx = hData[2];
      if (childIdx === -1) continue;

      let markerScore = (1.0 - Math.abs(1.0 - aspectRatio)) * extent;

      // Check for Grandchild (Square-in-Square-in-Square)
      const childHData = pHier.intPtr(0, childIdx);
      const grandchildIdx = childHData[2];

      if (grandchildIdx !== -1) {
        const grandchildArea = cv.contourArea(pContours.get(grandchildIdx));
        // Ensure grandchild is a meaningful part of the marker, not a speck
        if (grandchildArea > area * 0.05) {
          markerScore *= 3.0; // Huge boost for perfect signature
        }
      }

      candidates.push({ r, score: markerScore, index: i });
    }

    padded.delete();
    pContours.delete();
    pHier.delete();

    // Select the best 4 markers
    let rects: opencv.Rect[] = [];

    if (candidates.length >= 4) {
      // Sort by score and take top candidates
      candidates.sort((a, b) => b.score - a.score);
      const topCandidates = candidates.slice(0, 12); // Consider top 12 scorers

      // Heuristic: Pick 4 candidates that form the largest bounding box area
      // (This naturally picks the 4 corners of the page)
      let maxArea = -1;
      let bestCombo: opencv.Rect[] = [];

      for (let i = 0; i < topCandidates.length; i++) {
        for (let j = i + 1; j < topCandidates.length; j++) {
          for (let k = j + 1; k < topCandidates.length; k++) {
            for (let l = k + 1; l < topCandidates.length; l++) {
              const rs = [topCandidates[i].r, topCandidates[j].r, topCandidates[k].r, topCandidates[l].r];
              const minX = Math.min(...rs.map((r) => r.x));
              const maxX = Math.max(...rs.map((r) => r.x + r.width));
              const minY = Math.min(...rs.map((r) => r.y));
              const maxY = Math.max(...rs.map((r) => r.y + r.height));
              const comboArea = (maxX - minX) * (maxY - minY);

              // Also weight by the sum of scores
              const totalScore = rs.reduce((acc, _, idx) => acc + [topCandidates[i], topCandidates[j], topCandidates[k], topCandidates[l]][idx].score, 0);
              const combinedMetric = comboArea * totalScore;

              if (combinedMetric > maxArea) {
                maxArea = combinedMetric;
                bestCombo = rs;
              }
            }
          }
        }
      }
      rects = bestCombo;
    } else if (candidates.length === 3) {
      // ... (existing 3-marker recovery logic remains)
      const imgW = src.cols,
        imgH = src.rows;
      const idealCorners = [
        { name: "tl", x: 0, y: 0 },
        { name: "tr", x: imgW, y: 0 },
        { name: "br", x: imgW, y: imgH },
        { name: "bl", x: 0, y: imgH },
      ];
      const assigned: Record<string, opencv.Rect | null> = { tl: null, tr: null, br: null, bl: null };
      const usedCandidates = new Set();
      idealCorners.forEach((corner) => {
        let best: any = null;
        let minDist = Infinity;
        candidates.forEach((c, idx) => {
          if (usedCandidates.has(idx)) return;
          const cx = c.r.x + c.r.width / 2,
            cy = c.r.y + c.r.height / 2;
          const d = Math.sqrt((cx - corner.x) ** 2 + (cy - corner.y) ** 2);
          if (d < minDist) {
            minDist = d;
            best = { r: c.r, idx };
          }
        });
        if (best) {
          assigned[corner.name] = best.r;
          usedCandidates.add(best.idx);
        }
      });

      if (!assigned.tl && assigned.tr && assigned.br && assigned.bl) {
        assigned.tl = {
          x: assigned.tr!.x + assigned.bl!.x - assigned.br!.x,
          y: assigned.tr!.y + assigned.bl!.y - assigned.br!.y,
          width: assigned.br!.width,
          height: assigned.br!.height,
        } as any;
      } else if (!assigned.tr && assigned.tl && assigned.br && assigned.bl) {
        assigned.tr = {
          x: assigned.tl!.x + assigned.br!.x - assigned.bl!.x,
          y: assigned.tl!.y + assigned.br!.y - assigned.bl!.y,
          width: assigned.bl!.width,
          height: assigned.bl!.height,
        } as any;
      } else if (!assigned.br && assigned.tl && assigned.tr && assigned.bl) {
        assigned.br = {
          x: assigned.tr!.x + assigned.bl!.x - assigned.tl!.x,
          y: assigned.tr!.y + assigned.bl!.y - assigned.tl!.y,
          width: assigned.tl!.width,
          height: assigned.tl!.height,
        } as any;
      } else if (!assigned.bl && assigned.tl && assigned.tr && assigned.br) {
        assigned.bl = {
          x: assigned.tl!.x + assigned.br!.x - assigned.tr!.x,
          y: assigned.tl!.y + assigned.br!.y - assigned.tr!.y,
          width: assigned.tr!.width,
          height: assigned.tr!.height,
        } as any;
      }
      rects = Object.values(assigned).filter(Boolean) as opencv.Rect[];
    } else {
      rects = candidates.map((c) => c.r);
    }

    // Visual Stage: Anchor Detection (Drawn on Canvas via decorator to keep 'src' pure)
    await snap("corners", contrast, "Geometric Anchors", 700, (ctx, c) => {
      const scaleX = c.width / src.cols;
      const scaleY = c.height / src.rows;
      ctx.strokeStyle = "#0ea5e9";
      ctx.lineWidth = 8;

      rects.forEach((r) => {
        ctx.strokeRect(r.x * scaleX, r.y * scaleY, r.width * scaleX, r.height * scaleY);
      });

      if (rects.length === 4) {
        const tl = rects.reduce((a, b) => (a.x + a.y < b.x + b.y ? a : b));
        const tr = rects.reduce((a, b) => (a.x - a.y > b.x - b.y ? a : b));
        const br = rects.reduce((a, b) => (a.x + a.y > b.x + b.y ? a : b));
        const bl = rects.reduce((a, b) => (a.x - a.y < b.x - b.y ? a : b));
        const pts = [
          { x: (tl.x + tl.width / 2) * scaleX, y: (tl.y + tl.height / 2) * scaleY },
          { x: (tr.x + tr.width / 2) * scaleX, y: (tr.y + tr.height / 2) * scaleY },
          { x: (br.x + br.width / 2) * scaleX, y: (br.y + br.height / 2) * scaleY },
          { x: (bl.x + bl.width / 2) * scaleX, y: (bl.y + bl.height / 2) * scaleY },
        ];
        ctx.beginPath();
        ctx.lineWidth = 10;
        ctx.globalAlpha = 0.8;
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i <= 4; i++) {
          ctx.lineTo(pts[i % 4].x, pts[i % 4].y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    });

    // REAL PROCESSING: Create the final clean warped mat
    let warped = src.clone();

    // console.log(rects);

    // if (rects.length !== 4) return;

    if (rects.length === 4) {
      const tl = rects.reduce((a, b) => (a.x + a.y < b.x + b.y ? a : b));
      const tr = rects.reduce((a, b) => (a.x - a.y > b.x - b.y ? a : b));
      const br = rects.reduce((a, b) => (a.x + a.y > b.x + b.y ? a : b));
      const bl = rects.reduce((a, b) => (a.x - a.y < b.x - b.y ? a : b));

      // Point 0 (TL): Top-left of the tl-rect -> (tl.x, tl.y)
      // Point 1 (TR): Top-right of the tr-rect -> (tr.x + tr.width, tr.y)
      // Point 2 (BR): Bottom-right of the br-rect -> (br.x + br.width, br.y + br.height)
      // Point 3 (BL): Bottom-left of the bl-rect -> (bl.x, bl.y + bl.height)
      const s = cv.matFromArray(4, 1, cv.CV_32FC2, [
        tl.x,
        tl.y + tl.height,
        tr.x + tr.width,
        tr.y + tr.height,
        br.x + br.width,
        br.y,
        bl.x,
        bl.y,

        // tl.x,
        // tl.y,
        // tr.x + tr.width,
        // tr.y,
        // br.x + br.width,
        // br.y + br.height,
        // bl.x,
        // bl.y + bl.height
      ]);
      const d = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, 2400, 0, 2400, 3000, 0, 3000]);
      const M = cv.getPerspectiveTransform(s, d);
      warped = new cv.Mat();
      cv.warpPerspective(contrast, warped, M, new cv.Size(2400, 3000));
      M.delete();
      s.delete();
      d.delete();
    }
    await snap("warped", warped, "Perspective Correction");

    // Visual Stage: Grid Mapping / Text Block Analysis (Drawn on Canvas via decorator)
    await snap("blocks", warped, "Segment Mapping", 800, (ctx, c) => {
      const stepW = c.width / 9;
      const stepH = c.height / 11;
      ctx.strokeStyle = "#0ea5e9";
      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.9;
      for (let i = 1; i < 11; i++) {
        ctx.beginPath();
        ctx.moveTo(i * stepW, 0);
        ctx.lineTo(i * stepW, c.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * stepH);
        ctx.lineTo(c.width, i * stepH);
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    });

    setCurrentStep("extracting");
    const DIM = { rows: 11, cols: 9, rGap: 0, cGap: 0 };

    console.log("LOG 1");

    const w = (warped.cols - (DIM.cols + 1) * DIM.cGap) / DIM.cols;
    const h = (warped.rows - (DIM.rows + 1) * DIM.rGap) / DIM.rows;
    let idx = 0;

    console.log("LOG 2");

    const extracted: AdjustableGlyph[] = [];

    for (let r = 0; r < DIM.rows; r++) {
      for (let c = 0; c < DIM.cols; c++) {
        const char = TYPE_SEQUENCE[idx++];

        if (char) {
          // const x = (c + 1) * DIM.cGap + c * w;
          // const y = (r + 1) * DIM.rGap + r * h;

          const x = c * w;
          const y = r * h;
          const cc = document.createElement("canvas");
          cc.width = Math.round(w * 2.5);
          cc.height = Math.round(h * 2.5);

          console.log(x, y, w, h, char);

          const roi = warped.roi(new cv.Rect(x > 40 ? x - 40 : x, y + 60, w, h - 60));
          const tmpMat = new cv.Mat();
          cv.resize(roi, tmpMat, new cv.Size(cc.width, cc.height), 0, 0, cv.INTER_LANCZOS4);
          cv.imshow(cc, tmpMat);
          roi.delete();
          tmpMat.delete();

          const originalImg = cc.toDataURL("image/png");
          const g: AdjustableGlyph = {
            character: char,
            originalImg,
            xOffset: 0,
            yOffset: 0,
            scale: 1,
            rotation: 0,
          };
          extracted.push(g);
          setAdjustableGlyphs((p) => [...p, g]);
          await new Promise((r) => setTimeout(r, 10));
        }
      }
    }

    console.log("LOG 3");

    setCurrentStep("adjusting");
    setIsProcessing(false);
    gray.delete();
    thresh.delete();
    warped.delete();
    blurred.delete();
    contrast.delete();
    src.delete();
    console.log("LOG 4");

    setTimeout(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }, 400);
  };

  const generateFont = async (silent = false) => {
    if (adjustableGlyphs.length === 0) return;
    setIsGeneratingFont(true);
    setGenerationProgress(0);
    setGlyphs([]);

    const extracted: Glyph[] = [];

    for (let i = 0; i < adjustableGlyphs.length; i++) {
      const ag = adjustableGlyphs[i];
      setGenerationProgress(Math.round((i / adjustableGlyphs.length) * 100));
      // 1. Create a canvas to apply adjustments correctly
      const canvas = document.createElement("canvas");
      const img = new Image();
      img.src = ag.originalImg;
      await new Promise((r) => (img.onload = r));

      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;

      // 2. Apply adjustments (Background must be white for Potrace)
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      // Move to center, and apply percentage-based offsets relative to canvas size
      ctx.translate(canvas.width / 2 + (ag.xOffset / 100) * canvas.width, canvas.height / 2 + (ag.yOffset / 100) * canvas.height);
      ctx.rotate((ag.rotation * Math.PI) / 180);
      ctx.scale(ag.scale, ag.scale);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      ctx.restore();

      // 2b. Crop a 4% margin to remove cell-border grid artifacts
      const MARGIN = Math.round(canvas.width * 0.04);
      const innerW = canvas.width - MARGIN * 2;
      const innerH = canvas.height - MARGIN * 2;
      const croppedCanvas = document.createElement("canvas");
      croppedCanvas.width = innerW;
      croppedCanvas.height = innerH;
      croppedCanvas.getContext("2d")!.drawImage(canvas, MARGIN, MARGIN, innerW, innerH, 0, 0, innerW, innerH);

      // 3. Trace with OpenCV contours — Otsu threshold + RETR_CCOMP hierarchy
      // gives outer/hole classification for free, smooth bezier curves, no Potrace needed.
      const cleanSvg = cv ? traceGlyphWithOpenCV(cv, croppedCanvas, boldness, fidelity) : "";
      const adjustedImg = croppedCanvas.toDataURL("image/png");

      if (cleanSvg) {
        const g = { type: ag.character, svg: cleanSvg, originalImg: adjustedImg };
        extracted.push(g);
        setGlyphs((p) => [...p, g]);
      }
    }

    setGenerationProgress(100);
    setCurrentStep("done");
    setIsGeneratingFont(false);
    setTimeout(() => {
      applyGeneratedFont(extracted, silent);
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }, 400);
  };

  const updateAdjustableGlyph = (index: number, updates: Partial<AdjustableGlyph>) => {
    setAdjustableGlyphs((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], ...updates };
      return next;
    });
  };

  const getFontBuffer = (gs: Glyph[]): ArrayBuffer | null => {
    if (!gs.length) return null;
    const ogs: opentype.Glyph[] = [new opentype.Glyph({ name: ".notdef", unicode: 0, advanceWidth: 600, path: new opentype.Path() })];
    gs.forEach((g) => {
      const doc = new DOMParser().parseFromString(g.svg, "image/svg+xml");
      const rawD = Array.from(doc.getElementsByTagName("path"))
        .map((p) => p.getAttribute("d") || "")
        .filter(Boolean)
        .join(" ");
      if (!rawD) return;
      const svgEl = doc.getElementsByTagName("svg")[0];
      const vb = svgEl?.getAttribute("viewBox")?.split(/\s+/).map(Number);
      const svgH = vb && vb.length === 4 ? vb[3] : 100;
      const scale = (1000 / svgH) * globalScale;
      let minX = Infinity,
        maxX = -Infinity;
      svgpath(rawD)
        .abs()
        .iterate((seg: any) => {
          [seg[1], seg[3], seg[5]]
            .filter((v: any) => typeof v === "number")
            .forEach((x: number) => {
              minX = Math.min(minX, x);
              maxX = Math.max(maxX, x);
            });
        });
      if (minX === Infinity) return;

      // Absolutize entire path first so relative `m` hole sub-paths resolve correctly
      const absD = svgpath(rawD).abs().toString();

      // Winding: our OpenCV tracer already enforces the correct winding in image space:
      //   outer contours = CCW in Y-down image space
      //   hole  contours = CW  in Y-down image space
      // After scale(1, -scale) Y-flip:
      //   outer CCW → CW  (area < 0 in Y-up) = correct OpenType outer
      //   hole  CW  → CCW (area > 0 in Y-up) = correct OpenType hole
      // So we just apply the transform and emit — measure pre-flip area to guard
      // against any edge cases and reverse only if winding came out wrong.
      const path = new opentype.Path();
      splitSubPaths(absD).forEach((subD) => {
        const preSegs: any[] = [];
        svgpath(subD)
          .abs()
          .translate(-minX, 0)
          .iterate((seg: any) => preSegs.push([...seg]));
        if (!preSegs.length) return;
        const preArea = signedArea(flattenSegments(preSegs));
        if (preArea === 0) return;

        const transformed = svgpath(subD)
          .abs()
          .translate(-minX, 0)
          .scale(scale, -scale)
          .skewX(slant)
          .translate(0, 800 + globalYOffset);
        const segs: any[] = [];
        transformed.iterate((seg: any) => segs.push([...seg]));
        if (!segs.length) return;
        const postArea = signedArea(flattenSegments(segs));

        // outer in Y-down: CCW → preArea > 0 → needs CW post-flip → postArea < 0
        const needsCW = preArea > 0;
        const isCW = postArea < 0;
        const final = needsCW !== isCW ? reverseContour(segs) : segs;
        emitToPath(path, final);
      });
      ogs.push(new opentype.Glyph({ name: g.type, unicode: g.type.charCodeAt(0), advanceWidth: Math.round((maxX - minX) * scale + letterSpacing), path }));
    });

    // Add standard space character
    ogs.push(new opentype.Glyph({ name: "space", unicode: 32, advanceWidth: wordSpacing, path: new opentype.Path() }));

    try {
      return new opentype.Font({
        familyName: familyName || "MyHandwriting",
        styleName: styleName || "Regular",
        unitsPerEm: 1000,
        ascender: 800,
        descender: -200,
        designer: designer,
        version: version,
        description: description,
        glyphs: ogs,
      }).toArrayBuffer();
    } catch (e) {
      console.error(e);
      return null;
    }
  };

  const applyGeneratedFont = async (gs: Glyph[], silent = false) => {
    const buf = getFontBuffer(gs);
    if (!buf) return;

    if (!silent) {
      // Start audio early to compensate for latency
      const audio = new Audio(`${import.meta.env.BASE_URL}confetti-gun.mp3`);
      audio.play().catch(() => {});
    }

    const name = `UF_${Math.random().toString(36).substr(2, 8)}`;
    try {
      const f = await new FontFace(name, buf).load();
      document.fonts.add(f);
      setAppliedFontName(name);

      if (!silent) {
        // ── Success Celebration: Sequential Blasts ───────────────────────
        const colors = ["#0ea5e9", "#6366f1", "#8b5cf6"];

        // 1. Right Blast
        confetti({
          particleCount: 80,
          angle: 120,
          spread: 70,
          origin: { x: 1, y: 0.6 },
          colors: colors,
        });

        // 2. Left Blast after 300ms
        setTimeout(() => {
          confetti({
            particleCount: 80,
            angle: 60,
            spread: 70,
            origin: { x: 0, y: 0.6 },
            colors: colors,
          });
        }, 300);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const downloadTTF = () => {
    const buf = getFontBuffer(glyphs);
    if (!buf) return;
    const baseName = familyName.replace(/\s+/g, "_") || "MyHandwriting";
    const variant = styleName.replace(/\s+/g, "_") || "Regular";
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([buf], { type: "font/ttf" })),
      download: `${baseName}-${variant}.ttf`,
    });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  // ── Derived state helpers ────────────────────────────────────────────
  const isDone = currentStep === "done";
  const isIdle = currentStep === "idle";
  const isAdjusting = currentStep === "adjusting";

  return (
    <div className="min-h-screen flex flex-col antialiased text-slate-800" style={{ background: "#f8f9fb", fontFamily: "'Inter', sans-serif" }}>
      {/* ── Topbar ──────────────────────────────────────────────────────── */}
      <header className="h-14 px-6 bg-white/95 backdrop-blur border-b border-slate-100 flex items-center justify-between sticky top-0 z-50 shadow-sm shadow-slate-100/80">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-400 to-indigo-500 flex items-center justify-center shadow-md shadow-sky-500/30">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
              />
            </svg>
          </div>
          <span className="font-black text-slate-900 text-[15px] tracking-tight whitespace-nowrap">Handwriting Font</span>
          {isDone && (
            <span className="hidden sm:inline-flex items-center gap-1 text-[9px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              {glyphs.length} glyphs active
            </span>
          )}
        </div>

        {/* Progress steps — visible during processing */}
        {(isProcessing || isGeneratingFont) && (
          <div className="hidden md:flex items-center gap-1 text-[9px] font-bold uppercase tracking-widest">
            {["grayscale", "denoise", "contrast", "threshold", "corners", "warped", "blocks", "extracting"].map((step, idx) => {
              const steps = ["grayscale", "denoise", "contrast", "threshold", "corners", "warped", "blocks", "extracting"];
              const currentIdx = steps.indexOf(currentStep);
              const done = idx < currentIdx;
              const active = currentStep === step;
              return (
                <div key={step} className="flex items-center gap-1">
                  <span
                    className={`px-1.5 py-0.5 rounded transition-all duration-300 ${active ? "bg-sky-500 text-white scale-110" : done ? "text-emerald-500" : "text-slate-300"}`}>
                    {step.slice(0, 3)}
                  </span>
                  {idx < 7 && <span className={`w-3 h-px ${done ? "bg-emerald-300" : "bg-slate-200"}`} />}
                </div>
              );
            })}
          </div>
        )}

        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1.5 text-xs font-medium ${cv ? "text-emerald-600" : "text-amber-500"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${cv ? "bg-emerald-500" : "bg-amber-400 animate-pulse"}`} />
            <span className="hidden sm:inline">{cv ? "Ready" : "Loading…"}</span>
          </div>
        </div>
      </header>

      <main className="max-w-[1280px] mx-auto w-full px-6 py-12 flex flex-col gap-12">
        {/* ── Section 1: Hero ─────────────────────────────────────────────── */}
        <section className="relative pt-6 pb-4">
          {/* Ambient background blobs */}
          <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
            <div
              className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full opacity-[0.07]"
              style={{ background: "radial-gradient(circle, #0ea5e9, transparent 70%)" }}
            />
            <div
              className="absolute -bottom-20 -right-20 w-[400px] h-[400px] rounded-full opacity-[0.06]"
              style={{ background: "radial-gradient(circle, #6366f1, transparent 70%)" }}
            />
          </div>

          <div className="grid lg:grid-cols-[1fr_480px] gap-12 xl:gap-20 items-center">
            {/* ── Left: Copy ────────────────────────────────────────── */}
            <div className="flex flex-col gap-10">
              {/* Status pill */}
              <div className="flex items-center gap-2 w-fit">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-slate-200 bg-white shadow-sm text-[11px] font-semibold text-slate-500 tracking-wide">
                  <span className={`w-1.5 h-1.5 rounded-full ${cv ? "bg-emerald-400 shadow-[0_0_6px_#4ade80]" : "bg-amber-400 animate-pulse"}`} />
                  {cv ? "Engine ready" : "Initializing CV engine…"}
                </div>
              </div>

              {/* Headline */}
              <div className="flex flex-col gap-5">
                <h1
                  style={{ fontFamily: "'Georgia', serif", letterSpacing: "-0.03em" }}
                  className="text-[56px] xl:text-[68px] font-black text-slate-900 leading-[0.95]">
                  Turn your
                  <br />
                  <span className="relative inline-block">
                    <span
                      className="relative z-10"
                      style={{
                        backgroundImage: "linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        backgroundClip: "text",
                      }}>
                      handwriting
                    </span>
                    <span
                      className="absolute -bottom-1 left-0 w-full h-[3px] rounded-full opacity-30"
                      style={{ background: "linear-gradient(90deg, #0ea5e9, #6366f1)" }}
                    />
                  </span>
                  <br />
                  into a font.
                </h1>

                <p className="text-[17px] text-slate-500 leading-relaxed max-w-[420px]" style={{ fontFamily: "'Georgia', serif" }}>
                  Scan your characters, upload the grid, and get a professional <span className="text-slate-700 font-semibold">.TTF</span> file — powered by
                  computer vision in your browser.
                </p>
              </div>

              {/* CTA buttons */}
              <div className="flex flex-wrap gap-3">
                <a
                  href={`${import.meta.env.BASE_URL}FONT-TEMPLATE-NEW.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group inline-flex items-center gap-2.5 px-5 py-3 rounded-2xl bg-slate-900 text-white text-[13px] font-bold shadow-xl shadow-slate-900/20 hover:bg-slate-800 hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200">
                  <svg className="w-4 h-4 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.5"
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  Get Template
                </a>
                <a
                  href={`${import.meta.env.BASE_URL}demo-template.jpg`}
                  download
                  className="inline-flex items-center gap-2.5 px-5 py-3 rounded-2xl bg-white border border-slate-200 text-slate-600 text-[13px] font-bold shadow-sm hover:border-slate-300 hover:bg-slate-50 hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200">
                  <svg className="w-4 h-4 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Try Demo Scan
                </a>
              </div>

              {/* Feature grid — 3 compact items */}
              <div className="grid grid-cols-3 gap-3 pt-2 border-t border-slate-100">
                {[
                  { icon: "M13 10V3L4 14h7v7l9-11h-7z", label: "In-browser", sub: "No uploads to server" },
                  { icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z", label: "94 glyphs", sub: "Full ASCII coverage" },
                  { icon: "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4", label: "TTF export", sub: "OpenType compliant" },
                ].map(({ icon, label, sub }) => (
                  <div key={label} className="flex flex-col gap-1.5 p-3 rounded-xl bg-white border border-slate-100 shadow-sm">
                    <svg className="w-4 h-4 text-sky-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d={icon} />
                    </svg>
                    <p className="text-[12px] font-black text-slate-800">{label}</p>
                    <p className="text-[10px] text-slate-400 leading-tight">{sub}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Right: Upload card ─────────────────────────────────────── */}
            <div className="relative">
              {/* Glow halo */}
              <div
                className="absolute -inset-6 rounded-[40px] opacity-40 blur-3xl -z-10"
                style={{ background: "radial-gradient(ellipse at 60% 40%, #bae6fd 0%, #e0e7ff 60%, transparent 100%)" }}
              />

              <div className="bg-white rounded-3xl border border-slate-200/80 shadow-2xl shadow-slate-200/60 overflow-hidden">
                {/* Image zone */}
                <div className="relative bg-slate-50 group" style={{ aspectRatio: "4/3" }}>
                  {!imageFile ? (
                    <label
                      className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer transition-colors duration-300 hover:bg-sky-50/30"
                      style={{ backgroundImage: "radial-gradient(circle, #e2e8f0 1.5px, transparent 1.5px)", backgroundSize: "24px 24px" }}>
                      {/* Animated upload icon */}
                      <div className="relative mb-5">
                        <div className="w-16 h-16 rounded-2xl bg-white border border-slate-200 shadow-lg flex items-center justify-center group-hover:scale-110 group-hover:border-sky-300 group-hover:shadow-sky-200/60 transition-all duration-300">
                          <svg
                            className="w-7 h-7 text-slate-400 group-hover:text-sky-500 transition-colors duration-300"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24">
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                            />
                          </svg>
                        </div>
                        {/* Orbiting ring */}
                        <div className="absolute -inset-2 rounded-3xl border-2 border-dashed border-slate-200 group-hover:border-sky-300 group-hover:rotate-6 transition-all duration-500" />
                      </div>
                      <p className="text-[15px] font-bold text-slate-700 mb-1">Drop your scan here</p>
                      <p className="text-[12px] text-slate-400">
                        or <span className="text-sky-500 font-semibold underline underline-offset-2">browse files</span> · PNG / JPG
                      </p>
                      <p className="text-[10px] text-slate-300 mt-3 font-mono tracking-wider">9 × 11 CHARACTER GRID</p>
                      <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                    </label>
                  ) : (
                    <>
                      <img src={processedImage!} className="w-full h-full object-contain p-4" alt="scan" />

                      {isProcessing && (
                        <div className="absolute inset-0 bg-white/70 backdrop-blur-sm flex items-center justify-center">
                          <div className="absolute top-0 left-0 w-full h-[3px] bg-gradient-to-r from-sky-400 to-indigo-500 shadow-[0_0_12px_rgba(14,165,233,0.8)] animate-scan z-10" />
                          <div className="bg-white px-5 py-3 rounded-2xl shadow-2xl border border-slate-100 flex items-center gap-3">
                            <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
                            <span className="text-[11px] font-black text-slate-700 uppercase tracking-widest">{currentStep}…</span>
                          </div>
                        </div>
                      )}

                      {!isProcessing && (
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-6 gap-2.5">
                          <label className="px-4 py-2 bg-white/95 backdrop-blur text-slate-800 text-[12px] font-bold rounded-xl cursor-pointer hover:bg-white transition-all shadow-lg">
                            Change
                            <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                          </label>
                          {isIdle && (
                            <button
                              onClick={processImage}
                              className="px-6 py-2 bg-sky-500 text-white text-[12px] font-bold rounded-xl shadow-xl hover:bg-sky-600 active:scale-95 transition-all">
                              Run Pipeline
                            </button>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Action bar */}
                <div className="p-4 border-t border-slate-100 flex gap-2.5 bg-slate-50/50">
                  {!imageFile ? (
                    <label className="flex-1 py-3.5 rounded-2xl border-2 border-dashed border-slate-200 text-slate-400 text-[13px] font-bold flex items-center justify-center gap-2 cursor-pointer hover:border-sky-300 hover:text-sky-500 hover:bg-white transition-all">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4" />
                      </svg>
                      Select Scan
                      <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                    </label>
                  ) : isIdle ? (
                    <button
                      onClick={processImage}
                      disabled={!cv}
                      className="flex-1 py-3.5 rounded-2xl text-[13px] font-black flex items-center justify-center gap-2 active:scale-[0.98] disabled:opacity-40 transition-all shadow-lg shadow-sky-500/25 hover:-translate-y-0.5 hover:shadow-xl"
                      style={{ background: "linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%)", color: "white" }}>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Start Processing
                    </button>
                  ) : isAdjusting || isGeneratingFont ? (
                    <>
                      <button
                        onClick={() => generateFont()}
                        disabled={isGeneratingFont}
                        className="flex-1 py-3.5 rounded-2xl bg-slate-900 text-white text-[13px] font-black flex items-center justify-center gap-2 hover:bg-sky-600 active:scale-[0.98] disabled:opacity-40 transition-all shadow-xl">
                        {isGeneratingFont ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            Generating…
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            Generate Font
                          </>
                        )}
                      </button>
                      <div className="flex-1 py-3.5 rounded-2xl bg-white border border-slate-200 flex items-center justify-center gap-3 text-[13px] text-slate-500 shadow-sm">
                        <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
                        <span className="capitalize font-bold">{currentStep}…</span>
                      </div>
                    </>
                  ) : null}
                  {imageFile && (
                    <button
                      onClick={reset}
                      className="px-4 py-3.5 rounded-2xl border border-slate-200 text-slate-400 text-[13px] font-bold hover:border-rose-200 hover:text-rose-400 hover:bg-rose-50 active:scale-[0.98] transition-all">
                      ✕
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Section 2: Processing Pipeline ────────────────────────────── */}
        {(isProcessing || history.length > 1) && (
          <section className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden animate-fade-in-up">
            <div className="px-6 py-4 border-b border-slate-50 flex items-center justify-between">
              <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Processing Pipeline</h2>
              <div>
                {isProcessing && (
                  <span className="text-[11px] font-semibold text-sky-500 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-sky-500 rounded-full animate-pulse inline-block"></span>
                    {currentStep}
                  </span>
                )}
                {isDone && (
                  <span className="text-[11px] font-semibold text-emerald-500 flex items-center gap-1">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                    Complete — {glyphs.length} glyphs
                  </span>
                )}
              </div>
            </div>

            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
                {history.map((snap, i) => (
                  <div key={i} className="flex flex-col gap-4 group/stage cursor-default animate-fade-in-up" style={{ animationDelay: `${i * 100}ms` }}>
                    <div className="relative overflow-hidden rounded-2xl border border-slate-100 bg-slate-100 shadow-sm group-hover/stage:border-sky-200 group-hover/stage:shadow-xl group-hover/stage:-translate-y-1 transition-all duration-500">
                      <img src={snap.image} className="w-full aspect-[4/5] object-cover transition-all duration-700" alt={snap.name} />
                      {/* <div className="absolute top-4 left-4">
                        <span
                          className={`text-[10px] font-black px-2.5 py-1.5 rounded-lg text-white uppercase tracking-[0.1em] shadow-lg ${i === 0 ? "bg-slate-900/80" : "bg-sky-500/90"} backdrop-blur-md`}>
                          {i === 0 ? "Source Scan" : `Process ${i < 10 ? "0" : ""}${i}`}
                        </span>
                      </div> */}
                      <div className="absolute inset-0 ring-1 ring-inset ring-black/5 rounded-2xl"></div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest group-hover/stage:text-sky-500 transition-colors text-center">
                        {snap.name}
                      </span>
                      <div className="w-8 h-1 bg-slate-100 mt-2 rounded-full group-hover/stage:w-16 group-hover/stage:bg-sky-500 transition-all duration-500"></div>
                    </div>
                  </div>
                ))}
                {isProcessing &&
                  Array.from({ length: Math.max(0, 4 - (history.length % 4 || 4)) }).map((_, i) => (
                    <div key={i} className="flex flex-col gap-4 animate-pulse">
                      <div className="aspect-[4/5] rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/50 flex items-center justify-center">
                        <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">Neural Phase...</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <div className="h-3 w-32 bg-slate-100 rounded-full"></div>
                        <div className="w-8 h-1 bg-slate-50 mt-2 rounded-full"></div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </section>
        )}

        {/* ── Section 3: Adjustment Logic ────────────────────────────── */}
        {(isAdjusting || isGeneratingFont || isDone) && (
          <section className="flex flex-col gap-8 animate-fade-in-up scroll-mt-24" id="adjustment-section">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex flex-col gap-1">
                <h2 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-sky-500 flex items-center justify-center text-white text-sm shadow-lg shadow-sky-500/20">2</span>
                  {isDone ? "Refine Font Alignment" : "Fine-Tune Characters"}
                </h2>
                <p className="text-sm text-slate-500 font-medium">
                  {isDone
                    ? "Not satisfied with the result? Tweak any character and regenerate instantly."
                    : "Adjust size, alignment, and rotation for each character before vectorization."}
                </p>
              </div>
              <button
                onClick={() => generateFont()}
                disabled={isGeneratingFont}
                className={`px-8 py-4 ${isDone ? "bg-amber-500 hover:bg-amber-600" : "bg-slate-900 hover:bg-sky-600"} text-white text-sm font-bold rounded-2xl shadow-xl transition-all flex items-center gap-2.5 disabled:opacity-50`}>
                {isGeneratingFont ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Regenerating...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2.5"
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    {isDone ? "Update & Regenerate" : "Finish & Generate Font"}
                  </>
                )}
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {adjustableGlyphs.map((ag, i) => (
                <GlyphCard key={i} ag={ag} i={i} onUpdate={updateAdjustableGlyph} />
              ))}
            </div>

            <div className="flex justify-center pt-8 pb-12">
              <button
                onClick={() => generateFont()}
                disabled={isGeneratingFont}
                className="px-12 py-5 bg-sky-500 text-white text-base font-black rounded-2xl shadow-2xl shadow-sky-500/40 hover:bg-sky-600 hover:-translate-y-1 active:scale-95 transition-all flex items-center gap-3 disabled:opacity-50">
                {isGeneratingFont ? (
                  <>
                    <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                    Finalizing Glyph Vectors...
                  </>
                ) : (
                  <>
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    GENERATE CUSTOM FONT
                  </>
                )}
              </button>
            </div>
          </section>
        )}

        {/* ── Section 5: Global Refinement ────────────────────────────── */}
        {isDone && (
          <section className="relative bg-white rounded-3xl border-2 border-slate-900 shadow-2xl shadow-slate-200/50 overflow-hidden animate-fade-in-up">
            <div className="px-8 py-6 bg-slate-900 text-white flex items-center justify-between">
              <div className="flex flex-col gap-1">
                <h2 className="text-xl font-black tracking-tight flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-sky-500 flex items-center justify-center text-white text-sm shadow-lg shadow-sky-500/20">3</span>
                  Master Font Settings
                </h2>
                <p className="text-xs text-slate-400 font-medium">Fine-tune the overall size and baseline for the entire font.</p>
              </div>
              <button
                onClick={resetParams}
                className="group flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 hover:text-white text-[11px] font-bold transition-all border border-white/5 hover:border-white/20 active:scale-95">
                <svg
                  className="w-3.5 h-3.5 transition-transform duration-500 group-hover:rotate-[360deg]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="3"
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                Reset Defaults
              </button>
            </div>

            <div className="p-8 grid md:grid-cols-2 gap-12">
              <div className="flex flex-col gap-8">
                {/* Global Scale */}
                <div className="flex flex-col gap-4 p-5 bg-slate-50 rounded-2xl border border-slate-100">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-white shadow-sm flex items-center justify-center text-slate-400">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2.5"
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"
                          />
                        </svg>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-black text-slate-900">Global Font Size</span>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Master Scaling</span>
                      </div>
                    </div>
                    <span className="text-lg font-black text-sky-500 tabular-nums">{globalScale.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="4.0"
                    step="0.05"
                    value={globalScale}
                    onChange={(e) => setGlobalScale(parseFloat(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
                  />
                  <p className="text-[10px] text-slate-400 leading-relaxed italic border-l-2 border-slate-200 pl-3">
                    Increase this if your letters look small compared to standard fonts. Matches the average character height to the font em-square.
                  </p>
                </div>

                {/* Global Baseline */}
                <div className="flex flex-col gap-4 p-5 bg-slate-50 rounded-2xl border border-slate-100">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-white shadow-sm flex items-center justify-center text-slate-400">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
                        </svg>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-black text-slate-900">Vertical Alignment</span>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Baseline Offset</span>
                      </div>
                    </div>
                    <span className="text-lg font-black text-indigo-500 tabular-nums">
                      {globalYOffset > 0 ? "+" : ""}
                      {globalYOffset}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-400"
                    max="400"
                    step="10"
                    value={globalYOffset}
                    onChange={(e) => setGlobalYOffset(parseFloat(e.target.value))}
                    className="w-full h-2 rounded-full appearance-none cursor-pointer accent-indigo-500 bg-slate-200"
                  />
                </div>

                {/* Advanced Typographic Settings */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {/* Letter Spacing */}
                  <div className="flex flex-col gap-3 p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black text-slate-400 uppercase">Kerning</span>
                      <span className="text-xs font-bold text-sky-500">{letterSpacing}</span>
                    </div>
                    <input
                      type="range"
                      min="-50"
                      max="200"
                      step="5"
                      value={letterSpacing}
                      onChange={(e) => setLetterSpacing(parseInt(e.target.value))}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-100"
                    />
                    <p className="text-[8px] text-slate-400 uppercase font-bold tracking-tighter">Gap between letters</p>
                  </div>

                  {/* Word Spacing */}
                  <div className="flex flex-col gap-3 p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black text-slate-400 uppercase">Space Width</span>
                      <span className="text-xs font-bold text-indigo-500">{wordSpacing}</span>
                    </div>
                    <input
                      type="range"
                      min="100"
                      max="800"
                      step="20"
                      value={wordSpacing}
                      onChange={(e) => setWordSpacing(parseInt(e.target.value))}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-indigo-500 bg-slate-100"
                    />
                    <p className="text-[8px] text-slate-400 uppercase font-bold tracking-tighter">Gap between words</p>
                  </div>

                  {/* Slant */}
                  <div className="flex flex-col gap-3 p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black text-slate-400 uppercase">Slant (Italic)</span>
                      <span className="text-xs font-bold text-amber-500">{slant}°</span>
                    </div>
                    <input
                      type="range"
                      min="-25"
                      max="25"
                      step="1"
                      value={slant}
                      onChange={(e) => setSlant(parseInt(e.target.value))}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-amber-500 bg-slate-100"
                    />
                    <p className="text-[8px] text-slate-400 uppercase font-bold tracking-tighter">Mimic handwriting tilt</p>
                  </div>

                  {/* Ink Weight */}
                  <div className="flex flex-col gap-3 p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black text-slate-400 uppercase">Ink Weight</span>
                      <span className="text-xs font-bold text-rose-500">{boldness}</span>
                    </div>
                    <input
                      type="range"
                      min="-5"
                      max="12"
                      step="1"
                      value={boldness}
                      onChange={(e) => setBoldness(parseInt(e.target.value))}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-rose-500 bg-slate-100"
                    />
                    <p className="text-[8px] text-slate-400 uppercase font-bold tracking-tighter">Thicken or thin ink</p>
                  </div>

                  {/* Fidelity */}
                  <div className="flex flex-col gap-3 p-4 bg-white rounded-2xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black text-slate-400 uppercase">Fidelity</span>
                      <span className="text-xs font-bold text-emerald-500">{fidelity.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="15.0"
                      step="0.1"
                      value={fidelity}
                      onChange={(e) => setFidelity(parseFloat(e.target.value))}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-emerald-500 bg-slate-100"
                    />
                    <p className="text-[8px] text-slate-400 uppercase font-bold tracking-tighter">Smooth vs faithful trace</p>
                  </div>
                </div>

                {/* Silent Processing Progress Bar (Percentage) */}
                {isGeneratingFont && (
                  <div className="flex flex-col gap-2 animate-fade-in">
                    <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest">
                      <span className="text-slate-400">Rendering Optimized Glyphs...</span>
                      <span className="text-sky-500 tabular-nums">{generationProgress}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden border border-slate-50">
                      <div
                        className="h-full bg-gradient-to-r from-sky-400 to-indigo-500 transition-all duration-300 ease-out"
                        style={{ width: `${generationProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Font Metadata */}
                <div className="flex flex-col gap-6 p-6 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-sm font-black text-slate-900">Font Metadata</h3>
                      <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold">Embedded in .TTF file</p>
                    </div>
                  </div>

                  {/* Style Presets */}
                  <div className="flex flex-col gap-3">
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Quick Style Presets</label>
                    <div className="flex flex-wrap gap-2">
                      {[
                        { label: "Thin", b: -3, s: "Thin", color: "bg-slate-100 text-slate-600" },
                        { label: "Regular", b: 0, s: "Regular", color: "bg-sky-100 text-sky-700" },
                        { label: "Bold", b: 5, s: "Bold", color: "bg-indigo-100 text-indigo-700" },
                        { label: "Bolder", b: 10, s: "Extra-Bold", color: "bg-purple-100 text-purple-700" },
                        { label: "Italic", slant: 12, s: "Italic", color: "bg-amber-100 text-amber-700" },
                      ].map((p) => (
                        <button
                          key={p.label}
                          onClick={() => {
                            if (p.b !== undefined) setBoldness(p.b);
                            if (p.slant !== undefined) setSlant(p.slant);
                            setStyleName(p.s);
                          }}
                          className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-tight transition-all hover:scale-105 active:scale-95 ${p.color}`}>
                          {p.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Font Family Name *</label>
                      <input
                        type="text"
                        placeholder="e.g. My Handwriting"
                        className={`px-4 py-2.5 bg-white border ${!familyName ? "border-amber-300 ring-2 ring-amber-100" : "border-slate-200 shadow-sm"} rounded-xl text-sm focus:border-sky-500 focus:ring-4 focus:ring-sky-100 transition-all`}
                        value={familyName}
                        onChange={(e) => setFamilyName(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Style / Type</label>
                      <input
                        type="text"
                        placeholder="e.g. Handwriting"
                        className="px-4 py-2.5 bg-white border border-slate-200 shadow-sm rounded-xl text-sm focus:border-sky-500 focus:ring-4 focus:ring-sky-100 transition-all"
                        value={styleName}
                        onChange={(e) => setStyleName(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Designer / Author</label>
                      <input
                        type="text"
                        placeholder="e.g. John Doe"
                        className="px-4 py-2.5 bg-white border border-slate-200 shadow-sm rounded-xl text-sm focus:border-sky-500 focus:ring-4 focus:ring-sky-100 transition-all"
                        value={designer}
                        onChange={(e) => setDesigner(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Version</label>
                      <input
                        type="text"
                        placeholder="e.g. 1.000"
                        className="px-4 py-2.5 bg-white border border-slate-200 shadow-sm rounded-xl text-sm focus:border-sky-500 focus:ring-4 focus:ring-sky-100 transition-all"
                        value={version}
                        onChange={(e) => setVersion(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">Font Description</label>
                    <textarea
                      placeholder="About this font..."
                      rows={2}
                      className="px-4 py-2.5 bg-white border border-slate-200 shadow-sm rounded-xl text-sm focus:border-sky-500 focus:ring-4 focus:ring-sky-100 transition-all resize-none"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                    />
                  </div>
                </div>
                {/* 
                  Manual Update Button commented out per User Request. 
                  Auto-sync (Silent Update) is now handling all parameter changes.
                */}
                {/* 
                <button
                  onClick={() => applyGeneratedFont(glyphs)}
                  className="relative w-full py-4 px-6 overflow-hidden group bg-slate-900 hover:bg-slate-800 text-white rounded-2xl transition-all duration-300 shadow-xl shadow-slate-200 hover:shadow-sky-500/20 active:scale-[0.98] flex items-center justify-center gap-3">
                  <div className="absolute inset-0 bg-gradient-to-r from-sky-500 to-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  <div className="relative flex items-center gap-3">
                    <svg
                      className="w-5 h-5 transition-transform duration-700 group-hover:rotate-[360deg]"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2.5"
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    <span className="text-sm font-black uppercase tracking-wider">Update Font Preview</span>
                  </div>
                </button>
                */}
              </div>

              {/* Tips & Export Hint */}
              <div className="bg-slate-900 rounded-3xl p-8 text-slate-400 flex flex-col gap-6 relative overflow-hidden shadow-2xl">
                <div className="absolute top-0 right-0 w-48 h-48 bg-sky-500/20 rounded-full blur-[80px] -mr-24 -mt-24"></div>
                <div className="absolute bottom-0 left-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-[60px] -ml-16 -mb-16"></div>

                <div className="flex flex-col gap-2">
                  <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-sky-400">Export Final Font</h3>
                  <p className="text-sm leading-relaxed text-slate-300">
                    Your handwriting is currently set to <span className="text-white font-bold">{globalScale.toFixed(2)}x</span> scale.
                  </p>
                </div>

                {/* Settings Guide */}
                <div className="flex flex-col gap-5 py-4 border-y border-white/5">
                  <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">Typographic Guide</h3>
                  <div className="grid grid-cols-1 gap-5">
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Global Size:</strong> Recalibrates glyph height. Use this so your font
                        matches the size of standard system fonts when typed.
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Vertical Shift:</strong> Fixes "floating" text. Adjust the baseline offset
                        to ensure letters sit perfectly on the line.
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Kerning:</strong> Adjust if letters feel too far apart. Natural
                        handwriting usually looks better with a high kerning (+25 to +40).
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Space Width:</strong> Controls word distance. 300–400 is standard;
                        decrease for rapid, compact handwriting.
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Slant:</strong> Natural handwriting slants slightly to the right (+5° to
                        +12°). Programmatic italics.
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Ink Weight:</strong> Simulates different pens. Positive values mimic a
                        felt-tip marker; negative values mimic a fine-line pen.
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Fidelity:</strong> Lower values (0.1) capture every bump in the ink;
                        higher values (2.0+) create ultra-smooth, cleaned-up curves.
                      </p>
                    </div>
                    <div className="flex items-start gap-3 border-t border-white/5 pt-4">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 shrink-0" />
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        <strong className="text-slate-200 uppercase tracking-tighter">Metadata Branding:</strong> The Name, Designer, and Version are embedded
                        in the font. This is what you'll see in font menus like Microsoft Word or Apple Font Book.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-auto flex flex-col gap-4">
                  {!familyName && (
                    <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-amber-500"></div>
                      <p className="text-[10px] font-bold text-amber-500 uppercase">Font name is mandatory to export</p>
                    </div>
                  )}
                  <button
                    onClick={downloadTTF}
                    disabled={!familyName}
                    className="group w-full py-5 bg-white text-slate-900 font-black rounded-2xl hover:bg-sky-500 hover:text-white disabled:bg-slate-800 disabled:text-slate-600 transition-all shadow-xl flex items-center justify-center gap-3 active:scale-95">
                    <svg className="w-5 h-5 group-hover:rotate-12 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    DOWNLOAD .TTF FILE
                  </button>
                  <p className="text-[10px] text-center text-slate-500 font-medium">Compatible with Windows, macOS, and Mobile</p>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Section 6: Typography Playground ─────────────────────────── */}
        {isDone && (
          <section className="bg-white rounded-3xl border border-slate-100 shadow-xl overflow-hidden flex flex-col">
            {/* Toolbar */}
            <div className="px-5 py-4 border-b border-slate-100 bg-slate-50/30 flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
                  <h2 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest leading-none">Font Preview</h2>
                </div>
                {/* Show Comparison Toggle */}
                <label className="flex items-center gap-2 cursor-pointer bg-white px-3 py-1.5 rounded-full border border-slate-200 shadow-sm hover:border-sky-200 transition-all">
                  <div className="text-[9px] font-black text-slate-400 uppercase tracking-wider">Comparison Mode</div>
                  <input type="checkbox" checked={showComparison} onChange={(e) => setShowComparison(e.target.checked)} className="sr-only peer" />
                  <div className="w-7 h-4 bg-slate-200 peer-checked:bg-sky-500 rounded-full relative transition-colors duration-300 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:after:translate-x-3 shadow-inner"></div>
                </label>
              </div>
              <div className="flex items-center gap-3">
                {/* Font size control */}
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="text-[9px] font-black text-slate-400 uppercase tracking-wider">Size</span>
                  <input
                    type="range"
                    min="24"
                    max="160"
                    step="4"
                    value={previewFontSize}
                    onChange={(e) => setPreviewFontSize(parseInt(e.target.value))}
                    className="w-20 h-[3px] rounded-full appearance-none cursor-pointer accent-sky-500 bg-slate-200"
                  />
                  <span className="text-[9px] font-bold text-sky-500 w-8 tabular-nums">{previewFontSize}px</span>
                </div>
                <button
                  onClick={() => document.getElementById("adjustment-section")?.scrollIntoView({ behavior: "smooth" })}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 text-slate-500 text-xs font-semibold rounded-lg hover:border-sky-300 hover:text-sky-500 transition-all">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.5"
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                  Tune
                </button>
              </div>
            </div>

            {/* Main Area */}
            <div className={`p-8 grid gap-8 ${showComparison ? "lg:grid-cols-2 lg:divide-x lg:divide-slate-100" : "grid-cols-1"}`}>
              {/* Custom Font Block */}
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-black text-sky-500 uppercase tracking-widest bg-sky-50 px-2.5 py-1 rounded-lg">Custom Handwriting</span>
                  <span className="text-[10px] font-bold text-slate-400 tabular-nums">Font Active: {appliedFontName !== "inherit" ? "True" : "None"}</span>
                </div>
                <textarea
                  spellCheck="false"
                  className="w-full min-h-[360px] border-none focus:ring-0 p-0 text-slate-900 leading-[1.3] resize-none overflow-hidden placeholder:opacity-20 scrollbar-hide"
                  style={{ fontFamily: appliedFontName, fontSize: `${previewFontSize}px`, transition: "all 0.1s ease" }}
                  value={previewText}
                  onChange={(e) => setPreviewText(e.target.value)}
                  placeholder="Type your story here..."
                />
              </div>

              {/* Reference Font Block */}
              {showComparison && (
                <div className="flex flex-col gap-4 lg:pl-10">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-100 px-2.5 py-1 rounded-lg">Roboto Regular</span>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Global Bench</span>
                  </div>
                  <textarea
                    readOnly
                    spellCheck="false"
                    className="w-full min-h-[360px] border-none focus:ring-0 p-0 text-slate-900 leading-[1.3] resize-none overflow-hidden scrollbar-hide opacity-30 select-none"
                    style={{ fontFamily: "'Roboto', sans-serif", fontSize: `${previewFontSize}px` }}
                    value={previewText}
                  />
                </div>
              )}
            </div>

            {/* Metadata strip */}
            <div className="px-5 py-2.5 border-t border-slate-100 bg-slate-50/60 flex items-center gap-4 flex-wrap">
              <span className="text-[9px] text-slate-400 flex items-center gap-1">
                <span className="font-semibold text-slate-600">{glyphs.length}</span> vectorized
              </span>
              <span className="text-[9px] text-slate-300">·</span>
              <span className="text-[9px] text-slate-400 flex items-center gap-1">
                <span className="font-semibold text-slate-600">1000</span> units/EM
              </span>
              {appliedFontName !== "inherit" && (
                <>
                  <span className="text-[9px] text-slate-300">·</span>
                  <span className="text-[9px] font-mono text-sky-400 truncate max-w-[200px]">{appliedFontName}</span>
                </>
              )}
            </div>
          </section>
        )}

        {/* ── Section 7: Character Inventory (Final Output) ─────────────── */}
        {glyphs.length > 0 && isDone && (
          <section className="flex flex-col gap-4 animate-fade-in-up">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <h2 className="text-base font-bold text-slate-900 tracking-tight">Finalized Glyphs</h2>
                <span className="text-[10px] font-semibold text-emerald-600 bg-emerald-50 border border-emerald-100 px-2 py-0.5 rounded-full">
                  {glyphs.length} Vectorized
                </span>
              </div>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12 gap-2" ref={inventoryRef}>
              {glyphs.map((g, i) => (
                <div
                  key={i}
                  className="group bg-white rounded-xl border border-slate-100 p-2 flex flex-col gap-1.5 shadow-sm hover:shadow-md hover:border-emerald-200 hover:-translate-y-px transition-all duration-200 cursor-default">
                  <div className="aspect-square bg-slate-50 rounded-lg flex items-center justify-center p-2 overflow-hidden group-hover:bg-emerald-50/30 transition-colors">
                    <div
                      className="w-full h-full flex items-center justify-center grayscale group-hover:grayscale-0 group-hover:scale-110 transition-all duration-300"
                      dangerouslySetInnerHTML={{ __html: g.svg }}
                    />
                  </div>
                  <div className="flex items-center justify-between px-0.5">
                    <span className="text-[9px] font-bold text-slate-600">{g.type}</span>
                    <div className="w-3 h-3 rounded overflow-hidden opacity-0 group-hover:opacity-70 transition-opacity flex-shrink-0">
                      <img src={g.originalImg} className="w-full h-full object-contain" alt="" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="h-8"></div>
      </main>

      <footer className="w-full py-8 border-t border-slate-100 flex flex-col items-center gap-2">
        <p className="text-xs text-slate-400 font-medium">
          Made with Love by{" "}
          <a href="https://www.linkedin.com/in/manish-gun/" target="_blank" rel="noopener noreferrer" className="text-sky-500 font-bold hover:underline">
            Manish Gun
          </a>
        </p>
        <p className="text-[10px] text-slate-300">© 2025 All Rights Reserved</p>
      </footer>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

export default App;
