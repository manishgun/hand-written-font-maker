import { useEffect, useState, useRef } from "react";
import opencv from "@techstark/opencv-js";
import Potrace from "potrace";
import opentype from "opentype.js";
import { DOMParser } from "xmldom";
import svgpath from "svgpath";

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

function App() {
  const [cells, setCells] = useState<{ type: string; svg: string }[]>([]);
  const [cv, setCV] = useState<typeof opencv | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Check if OpenCV is already initialized
    opencv.onRuntimeInitialized = () => {
      setCV(opencv);
    };
  }, []);

  useEffect(() => {
    if (cv && canvasRef.current) {
      processImage();
    }
  }, [cv]);

  const processImage = () => {
    setIsProcessing(true);
    const img = new Image();
    // img.src = "/IMG_20260228_020759.jpg.jpeg";
    img.src = "/scan.jpg";
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Maintain aspect ratio while fitting in a reasonable size
      const maxDim = 1200;
      let width = img.width;
      let height = img.height;
      // if (width > maxDim || height > maxDim) {
      //   if (width > height) {
      //     height *= maxDim / width;
      //     width = maxDim;
      //   } else {
      //     width *= maxDim / height;
      //     height = maxDim;
      //   }
      // }

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);

      detectMarkers(canvas);
      setIsProcessing(false);
    };
    img.onerror = () => {
      console.error("Failed to load image");
      setIsProcessing(false);
    };
  };

  async function detectMarkers(canvas: HTMLCanvasElement) {
    if (!cv) return;
    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    const thresh = new cv.Mat();
    const edges = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    cv.imshow(canvas, gray);

    await new Promise((resolve) => {
      setTimeout(resolve, 500);
    });

    // cv.threshold(gray, thresh, 120, 255, cv.THRESH_BINARY_INV);

    // cv.imshow(canvas, thresh);

    // await new Promise((resolve) => {
    //   setTimeout(resolve, 500);
    // });

    cv.Canny(gray, edges, 50, 150);

    cv.imshow(canvas, edges);

    await new Promise((resolve) => {
      setTimeout(resolve, 500);
    });

    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const EDGE_POINTERS: opencv.Rect[] = [];

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const rect = cv.boundingRect(cnt);

        const aspect = rect.width / rect.height;

        if (rect.width > 25 && aspect > 0.8 && aspect < 1.4) {
          // console.log(rect, aspect);
          // cv.rectangle(src, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.width, rect.y + rect.height), new cv.Scalar(59, 130, 246, 255), 3);
          EDGE_POINTERS.push(rect);
        }
      }
      approx.delete();
    }

    cv.imshow(canvas, src);

    await new Promise((resolve) => {
      setTimeout(resolve, 500);
    });

    // if (EDGE_POINTERS.length) return;

    console.log(EDGE_POINTERS);

    if (EDGE_POINTERS.length === 4) {
      const tl = EDGE_POINTERS.reduce((a, b) => (a.x + a.y < b.x + b.y ? a : b));

      const tr = EDGE_POINTERS.reduce((a, b) => (a.x - a.y > b.x - b.y ? a : b));

      const br = EDGE_POINTERS.reduce((a, b) => (a.x + a.y > b.x + b.y ? a : b));

      const bl = EDGE_POINTERS.reduce((a, b) => (a.x - a.y < b.x - b.y ? a : b));

      const width = 800; // Math.floor(maxWidth);
      const height = 1000; // Math.floor(maxHeight);

      const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [tl.x, tl.y + tl.height, tr.x + tr.width, tr.y + tr.height, br.x + br.width, br.y, bl.x, bl.y]);

      const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width, 0, width, height, 0, height]);

      // 9️⃣ Perspective transform
      const M = cv.getPerspectiveTransform(srcTri, dstTri);
      const warped = new cv.Mat();

      cv.warpPerspective(src, warped, M, new cv.Size(width, height));

      // 10️⃣ Optional: crop inside margin
      // const margin = 40;
      // const cropped = warped.roi(new cv.Rect(margin, margin, width - 2 * margin, height - 2 * margin));

      // Show result
      cv.imshow(canvas, warped);
    }

    // cv.imshow(canvas, src);

    src.delete();
    gray.delete();
    thresh.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();

    cleanImage(canvas);
  }

  const cleanImage = (canvas: HTMLCanvasElement) => {
    if (!cv) return;

    // const src = cv.imread(canvas);
    // const gray = new cv.Mat();
    // const denoised = new cv.Mat();
    // const thresh = new cv.Mat();
    // const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));

    // 1️⃣ Grayscale
    // cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // 2️⃣ Strong noise reduction (better than Gaussian)
    // cv.medianBlur(gray, denoised, 5);

    // 3️⃣ Otsu threshold (automatic)
    // cv.threshold(denoised, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    // 4️⃣ Remove small black specks
    // cv.morphologyEx(gray, thresh, cv.MORPH_OPEN, kernel);

    // 5️⃣ Optional: Remove tiny connected components
    // removeSmallComponents(thresh, 50);

    // cv.imshow(canvas, src);

    // src.delete();
    // gray.delete();
    // denoised.delete();
    // kernel.delete();

    spliter(canvas);
  };

  const spliter = async (canvas: HTMLCanvasElement) => {
    const DIMENSIONS = {
      rows: 8,
      columns: 8,
      rowGap: 40,
      columnGap: 4,
    } as const;

    const cells: { svg: string; type: string }[] = [];

    const width = (canvas.width - (DIMENSIONS.columns + 1) * DIMENSIONS.columnGap) / DIMENSIONS.columns;
    const height = (canvas.height - (DIMENSIONS.rows + 1) * DIMENSIONS.rowGap) / DIMENSIONS.rows;

    const ctx = canvas.getContext("2d");

    let index = 0;

    if (ctx) {
      for (let r = 0; r < DIMENSIONS.rows; r++) {
        for (let c = 0; c < DIMENSIONS.columns; c++) {
          const TYPE = TYPE_SEQUENCE[index];

          if (TYPE) {
            const x = (c + 1) * DIMENSIONS.columnGap + c * width;
            const y = (r + 1) * DIMENSIONS.rowGap + r * height;

            const cellCanvas = document.createElement("canvas");

            cellCanvas.width = width;
            cellCanvas.height = height;

            const cellCtx = cellCanvas.getContext("2d");

            if (cellCtx) {
              cellCtx.drawImage(canvas, x, y + 2, width - 2, height - 2, 0, 0, width, height);
              //  const image =  cellCtx.getImageData(0, 0, width, height);

              const svg = await new Promise<string>((resolve, reject) => {
                Potrace.trace(cellCanvas.toDataURL(), (error, svg) => {
                  if (error) reject(error);
                  else if (svg) resolve(svg);
                });
              });

              cells.push({
                type: TYPE,
                svg: svg,
              });
            }
          }

          // const x = c * width;
          // const y = r * (height + DIMENSIONS.rowGap * r );

          index = index + 1;
        }
      }
    }

    setCells(cells);
  };

  const arrayToFont = (
    cells: {
      svg: string;
      type: string;
    }[],
  ) => {
    // Return early if no character cells have been processed
    if (cells.length === 0) return;

    // This array will hold the 'Glyph' objects (representations of characters) for our font
    const glyphs: opentype.Glyph[] = [];

    // 1. Add .notdef glyph: This is the fallback character shown when a font lacks a specific character
    glyphs.push(
      new opentype.Glyph({
        name: ".notdef",
        unicode: 0,
        advanceWidth: 600,
        path: new opentype.Path(),
      }),
    );

    // Filter out cells that don't contains actual SVG path data to avoid processing empty cells
    const validCells = cells.filter((c) => c.svg && c.svg.includes("<path"));

    for (let index = 0; index < validCells.length; index++) {
      const cell = validCells[index];
      // Use DOMParser to extract path and viewBox details from the raw SVG string
      const parser = new DOMParser();
      const doc = parser.parseFromString(cell.svg, "image/svg+xml");
      const pathElement = doc.getElementsByTagName("path")[0];
      const svgElement = doc.getElementsByTagName("svg")[0];
      const d = pathElement.getAttribute("d");

      // Skip this character if it has no path data
      if (!d) continue;

      let svgHeight = 100;
      // Get the height of the original SVG to calculate the scale needed for the font coordinate system
      const viewBox = svgElement.getAttribute("viewBox");
      if (viewBox) {
        const parts = viewBox.split(/\s+/).map(Number);
        if (parts.length === 4) svgHeight = parts[3];
      } else {
        const h = svgElement.getAttribute("height");
        if (h) svgHeight = parseFloat(h);
      }

      // Scale factor to map the handwriting SVG (height variable) to the font's Em square (800 units)
      const scale = 800 / svgHeight;

      // First pass: Find bounding box of the Raw SVG path to trim horizontal whitespace
      // We need this because scanned characters might have empty space on their left/right
      let minX = Infinity,
        maxX = -Infinity;
      let minY = Infinity,
        maxY = -Infinity;

      svgpath(d)
        .abs()
        .iterate((seg) => {
          const cmd = seg[0];
          // Track the boundaries (min/max X and Y) for all path segments (Lines, Curves, etc.)
          if (cmd === "M" || cmd === "L") {
            minX = Math.min(minX, seg[1]);
            maxX = Math.max(maxX, seg[1]);
            minY = Math.min(minY, seg[2]);
            maxY = Math.max(maxY, seg[2]);
          } else if (cmd === "C") {
            minX = Math.min(minX, seg[1], seg[3], seg[5]);
            maxX = Math.max(maxX, seg[1], seg[3], seg[5]);
            minY = Math.min(minY, seg[2], seg[4], seg[6]);
            maxY = Math.max(maxY, seg[2], seg[4], seg[6]);
          } else if (cmd === "Q") {
            minX = Math.min(minX, seg[1], seg[3]);
            maxX = Math.max(maxX, seg[1], seg[3]);
            minY = Math.min(minY, seg[2], seg[4]);
            maxY = Math.max(maxY, seg[2], seg[4]);
          }
        });

      // If no geometry found, skip
      if (minX === Infinity) continue;

      // Second pass: Transform the SVG path to Font Space
      // 1. Shift X so the character starts at 0 (removes left padding)
      // 2. Scale up (to match unitsPerEm) and flip Y (SVG uses top-down Y, Fonts use bottom-up Y)
      // 3. Move Y so it sits correctly relative to the baseline
      const transPath = svgpath(d)
        .abs()
        .translate(-minX, 0) // Trim left padding relative to character geometry
        .scale(scale, -scale) // Apply font scaling and vertical flip
        .translate(0, 800); // Position character above the baseline

      const path = new opentype.Path();
      let curX = 0,
        curY = 0;

      // Translate the transformed SVG path strings back into opentype.js Path operations
      transPath.iterate((seg) => {
        const cmd = seg[0];
        if (cmd === "M") {
          path.moveTo(seg[1], seg[2]);
          [curX, curY] = [seg[1], seg[2]];
        } else if (cmd === "L") {
          path.lineTo(seg[1], seg[2]);
          [curX, curY] = [seg[1], seg[2]];
        } else if (cmd === "C") {
          path.curveTo(seg[1], seg[2], seg[3], seg[4], seg[5], seg[6]);
          [curX, curY] = [seg[5], seg[6]];
        } else if (cmd === "Q") {
          path.quadTo(seg[1], seg[2], seg[3], seg[4]);
          [curX, curY] = [seg[3], seg[4]];
        } else if (cmd === "Z") {
          path.close();
        } else if (cmd === "H") {
          path.lineTo(seg[1], curY);
          curX = seg[1];
        } else if (cmd === "V") {
          path.lineTo(curX, seg[1]);
          curY = seg[1];
        }
      });

      // Calculate the specific advance width for this character based on its geometry
      const charWidth = (maxX - minX) * scale;
      glyphs.push(
        new opentype.Glyph({
          name: cell.type,
          unicode: cell.type.charCodeAt(0),
          advanceWidth: Math.round(charWidth + 80), // Set the spacing for the character
          path,
        }),
      );
    }

    // Assemble all processed glyphs into a complete Font object
    const font = new opentype.Font({
      familyName: "HandwrittenFont",
      styleName: "Regular",
      unitsPerEm: 1000,
      ascender: 800,
      descender: -200,
      glyphs,
    });

    // Generate the binary buffer of the font file
    const buffer = font.toArrayBuffer();
    // Create a Blob from the buffer and generate a URL for it to be accessible as a resource
    const blob = new Blob([buffer], { type: "font/ttf" });
    const url = URL.createObjectURL(blob);

    // Apply the newly generated font to the browser's document so the preview area updates immediately
    const fontFace = new FontFace("HandwrittenFont", buffer);
    fontFace.load().then((loadedFace) => {
      document.fonts.add(loadedFace);
      const previewText = document.getElementById("font-preview-text");
      if (previewText) {
        previewText.style.fontFamily = "HandwrittenFont";
      }
    });

    // Trigger an automatic download of the .ttf file for the user
    const a = document.createElement("a");
    a.href = url;
    a.download = "HandwrittenFont.ttf";
    a.click();
    // Clean up the URL object to free up memory
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black overflow-hidden selection:bg-blue-500/30">
      <div className="absolute top-0 left-0 w-full h-full opacity-20 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500 rounded-full blur-[120px]"></div>
      </div>

      <div className="z-10 w-full max-w-5xl flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
        <header className="flex flex-col gap-3 items-center text-center">
          <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400 text-xs font-semibold uppercase tracking-wider mb-2">
            AI Powered
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight bg-linear-to-r from-blue-400 via-indigo-400 to-emerald-400 bg-clip-text text-transparent sm:text-6xl">
            Font Sculptor
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
            Bring your handwriting to the digital world. Our intelligent marker detection identifies your custom characters with precision.
          </p>
        </header>

        <main className="relative group">
          <div className="absolute -inset-1 bg-linear-to-r from-blue-600/50 to-emerald-600/50 rounded-2xl blur-xl opacity-20 group-hover:opacity-40 transition duration-1000"></div>
          <div className="relative bg-slate-900/80 backdrop-blur-2xl border border-slate-800 rounded-2xl p-6 shadow-2xl overflow-hidden flex items-center justify-center min-h-[500px]">
            {(!cv || isProcessing) && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/40 backdrop-blur-md gap-6 transition-all duration-500">
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-blue-500/10 border-t-blue-500 rounded-full animate-spin"></div>
                  <div className="absolute inset-0 w-16 h-16 border-4 border-emerald-500/10 border-b-emerald-500 rounded-full animate-spin [animation-duration:1.5s]"></div>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <p className="text-white font-semibold text-lg tracking-wide">{!cv ? "Waking up the AI..." : "Extracting Details..."}</p>
                  <p className="text-slate-500 text-sm">{!cv ? "Loading OpenCV libraries" : "Analyzing document markers"}</p>
                </div>
              </div>
            )}

            <div className="w-full flex justify-center items-center">
              <canvas
                ref={canvasRef}
                className={`max-w-full h-auto rounded-lg shadow-2xl transition-all duration-700 ${isProcessing ? "scale-95 blur-sm opacity-50" : "scale-100 blur-0 opacity-100"}`}
              />
            </div>
          </div>

          <div className="grid grid-cols-8 gap-3">
            {cells.map((cell, idx) => {
              // return <img className="bg-white rounded" src={cell} key={idx} />;
              return <div className="bg-white rounded" dangerouslySetInnerHTML={{ __html: cell.svg }} key={idx} />;
            })}
          </div>
        </main>

        <footer className="flex flex-col items-center gap-6 mt-4">
          <div className="flex gap-4">
            <button
              onClick={processImage}
              disabled={!cv || isProcessing}
              className="px-10 py-4 bg-white text-slate-950 hover:bg-slate-100 font-bold rounded-2xl shadow-xl shadow-white/5 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Refresh Preview
            </button>

            <button
              onClick={() => arrayToFont(cells)}
              disabled={cells.length === 0}
              className="px-10 py-4 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-2xl shadow-xl shadow-emerald-500/20 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Font
            </button>
          </div>

          {cells.length > 0 && (
            <div className="w-full max-w-2xl bg-slate-900/50 border border-slate-800 rounded-xl p-6 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="flex items-center justify-between">
                <h3 className="text-slate-300 font-medium text-sm">Live Font Preview</h3>
                <span className="text-[10px] bg-slate-800 text-slate-500 px-2 py-0.5 rounded uppercase font-bold tracking-tighter">HandwrittenFont.ttf</span>
              </div>
              <textarea
                id="font-preview-text"
                placeholder="Type here to test your font..."
                defaultValue="The quick brown fox jumps over the lazy dog."
                className="w-full bg-transparent border-none text-4xl text-white resize-none focus:outline-none placeholder:text-slate-700 min-h-[100px]"
                style={{ fontFamily: "inherit" }}
              />
            </div>
          )}

          <div className="flex items-center gap-8 text-slate-500 text-sm">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              OpenCV WASM Ready
            </span>
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
              GPU Accelerated
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
