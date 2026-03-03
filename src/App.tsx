import { useEffect, useState, useRef } from "react";
import opencv from "@techstark/opencv-js";
import Potrace from "potrace";
import opentype from "opentype.js";
import { DOMParser } from "xmldom";
import svgpath from "svgpath";
import confetti from "canvas-confetti";

const TYPE_SEQUENCE = [
  "!",
  '"',
  "%",
  "&",
  "'",
  "(",
  undefined,
  undefined,
  ")",
  "+",
  ",",
  "-",
  ".",
  "/",
  undefined,
  undefined,
  ":",
  ";",
  "=",
  "?",
  "@",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
];

type Step = "idle" | "grayscale" | "denoise" | "contrast" | "threshold" | "edges" | "corners" | "warped" | "blocks" | "extracting" | "done";

interface Glyph {
  type: string;
  svg: string;
  originalImg: string;
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

function App() {
  const [cv, setCV] = useState<typeof opencv | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [currentStep, setCurrentStep] = useState<Step>("idle");
  const [isProcessing, setIsProcessing] = useState(false);
  const [glyphs, setGlyphs] = useState<Glyph[]>([]);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [history, setHistory] = useState<ProcessingSnapshot[]>([]);
  const [appliedFontName, setAppliedFontName] = useState<string>("inherit");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inventoryRef = useRef<HTMLDivElement>(null);

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
    setAppliedFontName("inherit");
  };

  const reset = () => {
    setImageFile(null);
    setGlyphs([]);
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
    blurred.delete();

    // Visual Stage: Contrast
    const contrast = new cv.Mat();
    gray.convertTo(contrast, -1, 1.4, 0);
    await snap("contrast", contrast, "Luma Correction");
    contrast.delete();

    const thresh = new cv.Mat();
    cv.threshold(gray, thresh, 120, 255, cv.THRESH_BINARY_INV);
    await snap("threshold", thresh, "Threshold");

    const edges = new cv.Mat();
    cv.Canny(gray, edges, 50, 150);
    await snap("edges", edges, "Edge Topology");

    const contours = new cv.MatVector(),
      hier = new cv.Mat();
    cv.findContours(edges, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const rects: opencv.Rect[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i),
        approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * cv.arcLength(cnt, true), true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const r = cv.boundingRect(cnt);
        if (r.width > 20 && r.width < 140) rects.push(r);
      }
      approx.delete();
    }

    // Visual Stage: Anchor Detection (Drawn on Canvas via decorator to keep 'src' pure)
    await snap("corners", src, "Geometric Anchors", 700, (ctx, c) => {
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
        ctx.lineWidth = 5;
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
    if (rects.length === 4) {
      const tl = rects.reduce((a, b) => (a.x + a.y < b.x + b.y ? a : b));
      const tr = rects.reduce((a, b) => (a.x - a.y > b.x - b.y ? a : b));
      const br = rects.reduce((a, b) => (a.x + a.y > b.x + b.y ? a : b));
      const bl = rects.reduce((a, b) => (a.x - a.y < b.x - b.y ? a : b));
      const s = cv.matFromArray(4, 1, cv.CV_32FC2, [tl.x, tl.y + tl.height, tr.x + tr.width, tr.y + tr.height, br.x + br.width, br.y, bl.x, bl.y]);
      const d = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, 800, 0, 800, 1000, 0, 1000]);
      const M = cv.getPerspectiveTransform(s, d);
      warped = new cv.Mat();
      cv.warpPerspective(src, warped, M, new cv.Size(800, 1000));
      M.delete();
      s.delete();
      d.delete();
    }
    await snap("warped", warped, "Perspective Correction");

    // Visual Stage: Grid Mapping / Text Block Analysis (Drawn on Canvas via decorator)
    await snap("blocks", warped, "Segment Mapping", 800, (ctx, c) => {
      const stepW = c.width / 8;
      const stepH = c.height / 8;
      ctx.strokeStyle = "#0ea5e9";
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.6;
      for (let i = 1; i < 8; i++) {
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
    const DIM = { rows: 8, cols: 8, rGap: 40, cGap: 14 };
    const w = (warped.cols - (DIM.cols + 1) * DIM.cGap) / DIM.cols;
    const h = (warped.rows - (DIM.rows + 1) * DIM.rGap) / DIM.rows;
    let idx = 0;
    const extracted: Glyph[] = [];

    for (let r = 0; r < DIM.rows; r++) {
      for (let c = 0; c < DIM.cols; c++) {
        const type = TYPE_SEQUENCE[idx++];
        if (type) {
          const x = (c + 1) * DIM.cGap + c * w,
            y = (r + 1) * DIM.rGap + r * h;
          const cc = document.createElement("canvas");
          cc.width = Math.round(w * 2.5);
          cc.height = Math.round(h * 2.5);

          // Extract directly from the warped Mat to avoid visual artifacts from the display canvas
          const roi = warped.roi(new cv.Rect(x, y + 2, w - 2, h - 2));
          const tmpMat = new cv.Mat();
          cv.resize(roi, tmpMat, new cv.Size(cc.width, cc.height), 0, 0, cv.INTER_LANCZOS4);
          cv.imshow(cc, tmpMat);
          roi.delete();
          tmpMat.delete();

          const originalImg = cc.toDataURL("image/png");
          const svg = await new Promise<string>((res) => Potrace.trace(originalImg, (_, r) => res(r || "")));
          if (svg) {
            const g = { type, svg, originalImg };
            extracted.push(g);
            setGlyphs((p) => [...p, g]);
            await new Promise((r) => setTimeout(r, 20));
          }
        }
      }
    }

    setCurrentStep("done");
    setIsProcessing(false);
    gray.delete();
    thresh.delete();
    edges.delete();
    contours.delete();
    hier.delete();
    warped.delete();
    src.delete();
    setTimeout(() => {
      applyGeneratedFont(extracted);
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }, 400);
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
      const scale = 800 / svgH;
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

      // ─── CRITICAL FIX ────────────────────────────────────────────────────
      // Potrace outputs hole sub-paths as relative `m x y` commands, e.g.:
      //   "M 100 200 C ... Z  m -50 -30 C ... Z"
      // The second `m` is relative to the PREVIOUS sub-path's end point.
      // Splitting rawD first and calling .abs() on each piece independently
      // resolves that `m` against (0,0), placing the hole at the wrong position.
      // Fix: absolutize the ENTIRE path string in one pass to resolve all
      // relative commands with the correct accumulated current position,
      // THEN split. Each piece now starts with an absolute `M`.
      const absD = svgpath(rawD).abs().toString();

      const path = new opentype.Path();
      splitSubPaths(absD).forEach((subD, subIdx) => {
        const transformed = svgpath(subD).abs().translate(-minX, 0).scale(scale, -scale).translate(0, 800);
        const segs: any[] = [];
        transformed.iterate((seg: any) => segs.push([...seg]));
        if (!segs.length) return;
        const area = signedArea(flattenSegments(segs));
        const isOuter = subIdx === 0;
        let final = segs;
        if (isOuter && area > 0) final = reverseContour(segs);
        else if (!isOuter && area < 0) final = reverseContour(segs);
        emitToPath(path, final);
      });
      ogs.push(new opentype.Glyph({ name: g.type, unicode: g.type.charCodeAt(0), advanceWidth: Math.round((maxX - minX) * scale + 60), path }));
    });
    try {
      return new opentype.Font({
        familyName: "CustomFont",
        styleName: "Regular",
        unitsPerEm: 1000,
        ascender: 800,
        descender: -200,
        glyphs: ogs,
      }).toArrayBuffer();
    } catch (e) {
      console.error(e);
      return null;
    }
  };

  const applyGeneratedFont = async (gs: Glyph[]) => {
    const buf = getFontBuffer(gs);
    if (!buf) return;

    // Start audio early to compensate for latency
    const audio = new Audio("/confetti-gun.mp3");
    audio.play().catch(() => {});

    const name = `UF_${Math.random().toString(36).substr(2, 8)}`;
    try {
      const f = await new FontFace(name, buf).load();
      document.fonts.add(f);
      setAppliedFontName(name);

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
    } catch (e) {
      console.error(e);
    }
  };

  const downloadTTF = () => {
    const buf = getFontBuffer(glyphs);
    if (!buf) return;
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([buf], { type: "font/ttf" })),
      download: "custom-handwriting.ttf",
    });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  // ── Derived state helpers ────────────────────────────────────────────
  const isDone = currentStep === "done";
  const isIdle = currentStep === "idle";

  return (
    <div className="min-h-screen bg-slate-50 font-['Inter'] flex flex-col antialiased text-slate-800">
      {/* ── Topbar ──────────────────────────────────────────────────────── */}
      <header className="h-14 px-6 bg-white border-b border-slate-100 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-sky-500 flex items-center justify-center shadow-sm shadow-sky-500/40">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
              />
            </svg>
          </div>
          <span className="font-bold text-slate-900 text-[15px] tracking-tight">FontForge</span>
          <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">Studio</span>
        </div>
        <div className={`flex items-center gap-1.5 text-xs font-medium ${cv ? "text-emerald-600" : "text-amber-500"}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${cv ? "bg-emerald-500" : "bg-amber-400 animate-pulse"}`}></span>
          {cv ? "Engine ready" : "Initializing…"}
        </div>
      </header>

      <main className="max-w-[1280px] mx-auto w-full px-6 py-12 flex flex-col gap-12">
        {/* ── Section 1: Hero Section ──────────────────────────────────────────── */}
        <section className="grid lg:grid-cols-2 gap-16 items-center py-8">
          {/* Left Side: Hero Content */}
          <div className="flex flex-col gap-8">
            <div className="inline-flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-sky-50 border border-sky-100 text-sky-600 text-[11px] font-bold uppercase tracking-wider w-fit">
              <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse"></span>
              Neural Extraction Engine v2.0
            </div>

            <div className="flex flex-col gap-4">
              <h1 className="text-[52px] font-black tracking-tight text-slate-900 leading-[1.05]">
                Your Handwriting <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-indigo-600">→ Perfect Typeface</span>
              </h1>
              <p className="text-lg text-slate-500 leading-relaxed max-w-lg">
                Upload a scanned character grid and watch our pipeline vectorize your handwriting into a professional .TTF font in real-time.
              </p>
            </div>

            <div className="flex flex-col gap-5">
              <div className="flex items-center gap-4 text-sm font-medium text-slate-600">
                <div className="w-10 h-10 rounded-xl bg-white border border-slate-100 shadow-sm flex items-center justify-center text-sky-500">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <p className="text-slate-900 font-bold">O(1) Processing</p>
                  <p className="text-slate-400 text-xs text-nowrap">Instant vectorization & manifold correction</p>
                </div>
              </div>

              <div className="flex items-center gap-4 text-sm font-medium text-slate-600">
                <div className="w-10 h-10 rounded-xl bg-white border border-slate-100 shadow-sm flex items-center justify-center text-indigo-500">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.5"
                      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                    />
                  </svg>
                </div>
                <div>
                  <p className="text-slate-900 font-bold">Winding Consistency</p>
                  <p className="text-slate-400 text-xs text-nowrap">Auto-corrects path orientation for TTF compliance</p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side: Upload Interface */}
          <div className="w-full relative">
            <div className="absolute -inset-4 bg-gradient-to-tr from-sky-100/50 to-indigo-100/50 blur-3xl opacity-50 -z-10 rounded-[40px]"></div>
            <div className="bg-white border border-slate-200 rounded-3xl shadow-xl shadow-slate-200/50 overflow-hidden">
              {/* Drop-zone */}
              <div className="relative aspect-[4/3] bg-slate-50 group">
                {!imageFile ? (
                  <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-sky-50/40 transition-colors duration-300">
                    <div className="w-16 h-16 rounded-2xl bg-white border border-slate-200 flex items-center justify-center mb-4 shadow-sm group-hover:border-sky-300 group-hover:shadow-sky-100 group-hover:scale-105 transition-all duration-300">
                      <svg className="w-7 h-7 text-slate-400 group-hover:text-sky-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
                    <p className="text-base font-bold text-slate-700">
                      Drop scan or <span className="text-sky-500 underline decoration-2 underline-offset-4">browse</span>
                    </p>
                    <p className="text-xs text-slate-400 mt-2">PNG / JPG · 8×8 character grid</p>
                    <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                  </label>
                ) : (
                  <>
                    <img src={processedImage!} className="w-full h-full object-contain p-4" alt="scan" />

                    {/* Scanning overlay */}
                    {isProcessing && (
                      <div className="absolute inset-0 bg-white/60 backdrop-blur-[1px] flex items-center justify-center">
                        <div className="absolute top-0 left-0 w-full h-[3px] bg-sky-500 shadow-[0_0_20px_rgba(14,165,233,1)] animate-scan z-10"></div>
                        <div className="bg-white/95 px-6 py-3 rounded-2xl shadow-2xl border border-slate-100 flex items-center gap-3 scale-110">
                          <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
                          <span className="text-xs font-bold text-slate-800 uppercase tracking-widest">{currentStep}…</span>
                        </div>
                      </div>
                    )}

                    {/* Hover overlay when idle or done */}
                    {!isProcessing && (
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-8 gap-3">
                        <label className="px-6 py-2.5 bg-white/95 backdrop-blur text-slate-900 text-xs font-bold rounded-xl cursor-pointer hover:bg-white transition active:scale-95 shadow-lg">
                          Change Image
                          <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                        </label>
                        {isIdle && (
                          <button
                            onClick={processImage}
                            className="px-8 py-2.5 bg-sky-500 text-white text-xs font-bold rounded-xl shadow-xl hover:bg-sky-600 active:scale-95 transition">
                            Run Full Pipeline
                          </button>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Bottom action bar */}
              <div className="px-6 py-5 border-t border-slate-100 flex gap-3 bg-slate-50/40">
                {!imageFile ? (
                  <label className="flex-1 py-3.5 rounded-2xl border-2 border-dashed border-slate-200 text-slate-400 text-sm font-bold flex items-center justify-center gap-2.5 cursor-pointer hover:border-sky-300 hover:text-sky-500 hover:bg-white transition-all">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4" />
                    </svg>
                    Select Template Scan
                    <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                  </label>
                ) : isIdle ? (
                  <button
                    onClick={processImage}
                    disabled={!cv}
                    className="flex-1 py-3.5 rounded-2xl bg-sky-500 text-white text-sm font-bold flex items-center justify-center gap-2.5 hover:bg-sky-600 active:scale-[0.98] disabled:opacity-40 transition-all shadow-lg shadow-sky-500/25">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Initialize Neural Logic
                  </button>
                ) : isDone ? (
                  <button
                    onClick={downloadTTF}
                    className="flex-1 py-3.5 rounded-2xl bg-slate-900 text-white text-sm font-bold flex items-center justify-center gap-2.5 hover:bg-sky-600 active:scale-[0.98] transition-all shadow-xl">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Export Custom .TTF
                  </button>
                ) : (
                  <div className="flex-1 py-3.5 rounded-2xl bg-white border border-slate-200 flex items-center justify-center gap-3 text-sm text-slate-500 shadow-sm">
                    <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="capitalize font-bold tracking-wide">{currentStep}…</span>
                  </div>
                )}
                {imageFile && (
                  <button
                    onClick={reset}
                    className="px-6 py-3.5 rounded-2xl border border-slate-200 text-slate-400 text-sm font-bold hover:border-rose-200 hover:text-rose-500 hover:bg-rose-50 active:scale-[0.98] transition-all">
                    Reset
                  </button>
                )}
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
                      <div className="absolute top-4 left-4">
                        <span
                          className={`text-[10px] font-black px-2.5 py-1.5 rounded-lg text-white uppercase tracking-[0.1em] shadow-lg ${i === 0 ? "bg-slate-900/80" : "bg-sky-500/90"} backdrop-blur-md`}>
                          {i === 0 ? "Source Scan" : `Process ${i < 10 ? "0" : ""}${i}`}
                        </span>
                      </div>
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

        {/* ── Section 3: Character Inventory ────────────────────────────── */}
        {(glyphs.length > 0 || isProcessing) && (
          <section className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <h2 className="text-base font-bold text-slate-900 tracking-tight">Character Inventory</h2>
                {glyphs.length > 0 && (
                  <span className="text-[10px] font-semibold text-sky-600 bg-sky-50 border border-sky-100 px-2 py-0.5 rounded-full">
                    {glyphs.length} glyphs
                  </span>
                )}
              </div>
              {isProcessing && glyphs.length > 0 && (
                <span className="text-xs text-slate-400 font-medium flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 bg-sky-400 rounded-full animate-ping inline-block"></span>
                  Extracting…
                </span>
              )}
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12 gap-2" ref={inventoryRef}>
              {glyphs.map((g, i) => (
                <div
                  key={i}
                  className="group bg-white rounded-xl border border-slate-100 p-2 flex flex-col gap-1.5 shadow-sm hover:shadow-md hover:border-sky-200 hover:-translate-y-px transition-all duration-200 cursor-default">
                  <div className="aspect-square bg-slate-50 rounded-lg flex items-center justify-center p-2 overflow-hidden group-hover:bg-sky-50/30 transition-colors">
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
              {isProcessing &&
                glyphs.length === 0 &&
                Array.from({ length: 24 }).map((_, i) => <div key={i} className="aspect-square bg-white border border-slate-100 rounded-xl animate-pulse" />)}
            </div>
          </section>
        )}

        {/* ── Section 4: Typography Playground ─────────────────────────── */}
        {isDone && (
          <section className="bg-white rounded-2xl border border-sky-100 shadow-xl overflow-hidden flex flex-col">
            {/* Toolbar */}
            <div className="px-6 py-3.5 border-b border-slate-50 bg-slate-50/50 flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-3">
                <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Typography Playground</h2>
                <span className="inline-flex items-center gap-1 text-[9px] font-semibold text-emerald-600 bg-emerald-50 border border-emerald-100 px-2 py-0.5 rounded-full">
                  <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                  </svg>
                  Font Active
                </span>
              </div>
              <button
                onClick={downloadTTF}
                className="flex items-center gap-1.5 px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-lg hover:bg-sky-600 active:scale-95 transition-all">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export .TTF
              </button>
            </div>

            {/* Canvas area with dot grid */}
            <div
              className="relative flex-1 min-h-[340px] flex items-center justify-center p-10"
              style={{ backgroundImage: "radial-gradient(circle, #cbd5e1 1px, transparent 1px)", backgroundSize: "22px 22px" }}>
              <textarea
                id="font-preview"
                defaultValue="The quick brown fox jumps over the lazy dog."
                style={{ fontFamily: appliedFontName, fontSize: 100, lineHeight: 1.2 }}
                className="relative z-10 w-full text-center bg-transparent uppercase border-none focus:outline-none resize-none no-scrollbar text-slate-900 placeholder:text-slate-200"
                spellCheck={false}
                rows={3}
              />
            </div>

            {/* Metadata strip */}
            <div className="px-6 py-3 border-t border-slate-50 bg-slate-50/40 flex items-center gap-5 flex-wrap">
              <span className="text-[10px] text-slate-400">
                <span className="font-semibold text-slate-600">{glyphs.length}</span> glyphs
              </span>
              <span className="text-[10px] text-slate-400">
                <span className="font-semibold text-slate-600">1000</span> units/EM
              </span>
              <span className="text-[10px] font-mono text-sky-500 truncate max-w-[240px]">{appliedFontName}</span>
            </div>
          </section>
        )}

        <div className="h-8"></div>
      </main>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

export default App;
