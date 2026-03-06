import { useEffect, useState, useRef } from "react";
import opencv from "@techstark/opencv-js";
import Potrace from "potrace";
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
              const minX = Math.min(...rs.map(r => r.x));
              const maxX = Math.max(...rs.map(r => r.x + r.width));
              const minY = Math.min(...rs.map(r => r.y));
              const maxY = Math.max(...rs.map(r => r.y + r.height));
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
      const imgW = src.cols, imgH = src.rows;
      const idealCorners = [
        { name: 'tl', x: 0, y: 0 },
        { name: 'tr', x: imgW, y: 0 },
        { name: 'br', x: imgW, y: imgH },
        { name: 'bl', x: 0, y: imgH }
      ];
      const assigned: Record<string, opencv.Rect | null> = { tl: null, tr: null, br: null, bl: null };
      const usedCandidates = new Set();
      idealCorners.forEach(corner => {
        let best: any = null;
        let minDist = Infinity;
        candidates.forEach((c, idx) => {
          if (usedCandidates.has(idx)) return;
          const cx = c.r.x + c.r.width / 2, cy = c.r.y + c.r.height / 2;
          const d = Math.sqrt((cx - corner.x) ** 2 + (cy - corner.y) ** 2);
          if (d < minDist) { minDist = d; best = { r: c.r, idx }; }
        });
        if (best) { assigned[corner.name] = best.r; usedCandidates.add(best.idx); }
      });

      if (!assigned.tl && assigned.tr && assigned.br && assigned.bl) {
        assigned.tl = { x: assigned.tr!.x + assigned.bl!.x - assigned.br!.x, y: assigned.tr!.y + assigned.bl!.y - assigned.br!.y, width: assigned.br!.width, height: assigned.br!.height } as any;
      } else if (!assigned.tr && assigned.tl && assigned.br && assigned.bl) {
        assigned.tr = { x: assigned.tl!.x + assigned.br!.x - assigned.bl!.x, y: assigned.tl!.y + assigned.br!.y - assigned.bl!.y, width: assigned.bl!.width, height: assigned.bl!.height } as any;
      } else if (!assigned.br && assigned.tl && assigned.tr && assigned.bl) {
        assigned.br = { x: assigned.tr!.x + assigned.bl!.x - assigned.tl!.x, y: assigned.tr!.y + assigned.bl!.y - assigned.tl!.y, width: assigned.tl!.width, height: assigned.tl!.height } as any;
      } else if (!assigned.bl && assigned.tl && assigned.tr && assigned.br) {
        assigned.bl = { x: assigned.tl!.x + assigned.br!.x - assigned.tr!.x, y: assigned.tl!.y + assigned.br!.y - assigned.tr!.y, width: assigned.tr!.width, height: assigned.tr!.height } as any;
      }
      rects = Object.values(assigned).filter(Boolean) as opencv.Rect[];
    } else {
      rects = candidates.map(c => c.r);
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

    console.log(rects);

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
        tl.x, tl.y + tl.height,
        tr.x + tr.width, tr.y + tr.height,
        br.x + br.width, br.y,
        bl.x, bl.y




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
    const w = (warped.cols - (DIM.cols + 1) * DIM.cGap) / DIM.cols;
    const h = (warped.rows - (DIM.rows + 1) * DIM.rGap) / DIM.rows;
    let idx = 0;
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

          const roi = warped.roi(new cv.Rect(x + 4, y + 50, w - 4, h - 4));
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

    setCurrentStep("adjusting");
    setIsProcessing(false);
    gray.delete();
    thresh.delete();
    warped.delete();
    blurred.delete();
    contrast.delete();
    src.delete();
    setTimeout(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }, 400);
  };

  const generateFont = async () => {
    if (adjustableGlyphs.length === 0) return;
    setIsGeneratingFont(true);
    setGlyphs([]);

    const extracted: Glyph[] = [];

    for (const ag of adjustableGlyphs) {
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

      const adjustedImg = canvas.toDataURL("image/png");

      // 3. Trace SVG from the ADJUSTED image
      const svg = await new Promise<string>((res) => Potrace.trace(adjustedImg, { turdSize: 160 }, (_, r) => res(r || "")));
      if (svg) {
        // Use adjustedImg for the inventory preview so user sees their work
        const g = { type: ag.character, svg, originalImg: adjustedImg };
        extracted.push(g);
        setGlyphs((p) => [...p, g]);
      }
    }

    setCurrentStep("done");
    setIsGeneratingFont(false);
    setTimeout(() => {
      applyGeneratedFont(extracted);
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
    const audio = new Audio(`${import.meta.env.BASE_URL}confetti-gun.mp3`);
    audio.play().catch(() => { });

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
  const isAdjusting = currentStep === "adjusting";

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
          <span className="font-bold text-slate-900 text-[15px] tracking-tight whitespace-nowrap">Hand Writing Font</span>
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
                <span className="text-transparent bg-clip-text bg-linear-to-r from-sky-500 to-indigo-600">→ Perfect Typeface</span>
              </h1>
              <p className="text-lg text-slate-500 leading-relaxed max-w-lg">
                Upload a scanned character grid and watch our pipeline vectorize your handwriting into a professional .TTF font in real-time.
              </p>

              <div className="flex flex-wrap gap-3 mt-2">
                <a
                  href={`${import.meta.env.BASE_URL}FONT-TEMPLATE.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 bg-sky-500 text-white text-xs font-bold rounded-xl shadow-lg shadow-sky-500/20 hover:bg-sky-600 transition-all flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.5"
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  Get Blank Template
                </a>
                <a
                  href={`${import.meta.env.BASE_URL}demo-template.jpg`}
                  download
                  className="px-5 py-2.5 bg-white border border-slate-200 text-slate-600 text-xs font-bold rounded-xl shadow-sm hover:border-sky-200 hover:text-sky-500 transition-all flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Demo Scan
                </a>
              </div>
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
                    <p className="text-xs text-slate-400 mt-2">PNG / JPG · 9×11 character grid</p>
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
                ) : isAdjusting || isGeneratingFont ? (
                  <button
                    onClick={generateFont}
                    disabled={isGeneratingFont}
                    className="flex-1 py-3.5 rounded-2xl bg-slate-900 text-white text-sm font-bold flex items-center justify-center gap-2.5 hover:bg-sky-600 active:scale-[0.98] disabled:opacity-40 transition-all shadow-xl">
                    {isGeneratingFont ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Generating...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Generate Font
                      </>
                    )}
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
                onClick={generateFont}
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

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {adjustableGlyphs.map((ag, i) => (
                <div
                  key={i}
                  className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex flex-col gap-4 hover:border-sky-200 transition-colors group">
                  <div className="flex items-center justify-between px-1">
                    <span className="text-lg font-black text-slate-900">{ag.character}</span>
                    <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">#{i + 1}</span>
                  </div>

                  <div className="aspect-square bg-slate-50 rounded-xl overflow-hidden flex items-center justify-center border border-slate-100 relative">
                    <img
                      src={ag.originalImg}
                      className="max-w-[70%] max-h-[70%] object-contain grayscale opacity-80 group-hover:opacity-100 transition-opacity"
                      style={{
                        transform: `translate(${ag.xOffset}%, ${ag.yOffset}%) scale(${ag.scale}) rotate(${ag.rotation}deg)`,
                        transition: "transform 0.1s ease-out",
                      }}
                    />
                    <div className="absolute inset-0 border border-slate-200/30 pointer-events-none"></div>

                    {/* 3x3 Grid Guide Lines */}
                    <div className="absolute top-[25%] left-0 w-full h-[0.5px] bg-sky-500/5 pointer-events-none"></div>
                    <div className="absolute top-[50%] left-0 w-full h-[0.5px] bg-sky-500/20 pointer-events-none"></div>
                    <div className="absolute top-[75%] left-0 w-full h-[0.5px] bg-sky-500/5 pointer-events-none"></div>

                    <div className="absolute left-[25%] top-0 w-[0.5px] h-full bg-sky-500/5 pointer-events-none"></div>
                    <div className="absolute left-[50%] top-0 w-[0.5px] h-full bg-sky-500/20 pointer-events-none"></div>
                    <div className="absolute left-[75%] top-0 w-[0.5px] h-full bg-sky-500/5 pointer-events-none"></div>
                  </div>

                  <div className="flex flex-col gap-4 bg-slate-50/50 p-3 rounded-xl">
                    <div className="flex flex-col gap-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[9px] font-black text-slate-400 uppercase tracking-wider">Scale</label>
                        <span className="text-[9px] font-bold text-sky-600">{ag.scale.toFixed(2)}x</span>
                      </div>
                      <input
                        type="range"
                        min="0.4"
                        max="2.5"
                        step="0.05"
                        value={ag.scale}
                        onChange={(e) => updateAdjustableGlyph(i, { scale: parseFloat(e.target.value) })}
                        className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
                      />
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[9px] font-black text-slate-400 uppercase tracking-wider">Rotation</label>
                        <span className="text-[9px] font-bold text-sky-600">{ag.rotation}°</span>
                      </div>
                      <input
                        type="range"
                        min="-45"
                        max="45"
                        step="1"
                        value={ag.rotation}
                        onChange={(e) => updateAdjustableGlyph(i, { rotation: parseFloat(e.target.value) })}
                        className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex flex-col gap-1.5">
                        <label className="text-[9px] font-black text-slate-400 uppercase tracking-wider">X Offset</label>
                        <input
                          type="range"
                          min="-50"
                          max="50"
                          step="1"
                          value={ag.xOffset}
                          onChange={(e) => updateAdjustableGlyph(i, { xOffset: parseFloat(e.target.value) })}
                          className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <label className="text-[9px] font-black text-slate-400 uppercase tracking-wider">Y Offset</label>
                        <input
                          type="range"
                          min="-50"
                          max="50"
                          step="1"
                          value={ag.yOffset}
                          onChange={(e) => updateAdjustableGlyph(i, { yOffset: parseFloat(e.target.value) })}
                          className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex justify-center pt-8 pb-12">
              <button
                onClick={generateFont}
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

        {/* ── Section 4: Character Inventory ────────────────────────────── */}
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
              <div className="flex items-center gap-2">
                <button
                  onClick={() => document.getElementById("adjustment-section")?.scrollIntoView({ behavior: "smooth" })}
                  className="flex items-center gap-1.5 px-4 py-2 bg-white border border-slate-200 text-slate-600 text-xs font-semibold rounded-lg hover:border-sky-300 hover:text-sky-500 transition-all">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.5"
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                  Tune Alignment
                </button>
                <button
                  onClick={downloadTTF}
                  className="flex items-center gap-1.5 px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-lg hover:bg-sky-600 active:scale-95 transition-all">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export .TTF
                </button>
              </div>
            </div>

            {/* Canvas area with dot grid */}
            <div
              className="relative flex-1 min-h-[340px] flex items-center justify-center p-10"
              style={{ backgroundImage: "radial-gradient(circle, #cbd5e1 1px, transparent 1px)", backgroundSize: "22px 22px" }}>
              <textarea
                id="font-preview"
                defaultValue="The quick brown fox jumps over the lazy dog."
                style={{ fontFamily: appliedFontName, fontSize: 100, lineHeight: 1.2 }}
                className="relative z-10 w-full text-center bg-transparent border-none focus:outline-none resize-none no-scrollbar text-slate-900 placeholder:text-slate-200"
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
