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
    img.src = "/IMG_20260228_020759.jpg.jpeg";
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

          if (TYPE === undefined) continue;

          // const x = c * width;
          // const y = r * (height + DIMENSIONS.rowGap * r );

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
              Potrace.trace(
                cellCanvas.toDataURL(),
                // {
                //   threshold: 100,
                //   turdSize: 6, // removes small noise
                //   optTolerance: 0.2,
                //   // turdPolicy: "minority",
                // },
                (error, svg) => {
                  if (error) reject(error);
                  else if (svg) resolve(svg);
                },
              );
            });

            const parser = new DOMParser();
            const doc = parser.parseFromString(svg, "image/svg+xml");

            const pathElement = doc.getElementsByTagName("path")[0];
            const d = pathElement.getAttribute("d");

            const element = doc.getElementsByTagName("svg")[0];

            if (element !== null) {
              const viewBox = element.getAttribute("viewBox").split(" ").map(Number);
            }

            cells.push({
              type: TYPE,
              svg: svg,
            });
            // cells.push(cellCanvas.toDataURL());
          }

          index = index + 1;
        }
      }
    }

    setCells(cells);
  };

  const arrayToFont = () => {};

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
              disabled={true}
              className="px-10 py-4 bg-slate-800 text-slate-400 font-bold rounded-2xl border border-slate-700 cursor-not-allowed opacity-50">
              Download Font (Coming Soon)
            </button>
          </div>

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
