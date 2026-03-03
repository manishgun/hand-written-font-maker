# Hand Writing Font 🖋️

![Hand Writing Font Thumbnail](https://via.placeholder.com/1280x640?text=Hand+Writing+Font+Preview)

**Hand Writing Font** is a high-performance, browser-based neural extraction engine that converts your handwritten character scans into professional, vector-based `.TTF` font files in real-time.

[**✨ Try the Live Demo: font.crudios.com**](https://font.crudios.com)

---

## 🚀 Key Features

- **🧠 Neural Processing Pipeline**: Built with OpenCV.js for high-fidelity image analysis.
- **📐 Auto-Perspective Correction**: Uses 4-corner anchor detection to de-warp and normalize document scans.
- **⚡ Advanced Vectorization**: Integrates Potrace for converting raster strokes into smooth, scalable SVG paths.
- **🛠️ TTF Compliance Engine**:
  - Auto-corrects winding order (Shoelace area algorithm).
  - Preserves native Bezier curves for professional edge smoothness.
  - Manages complex hole sub-paths for characters like 'O', 'B', and 'P'.
- **🎭 Interactive Typography Playground**: Test your generated font instantly with a real-time preview.
- **📦 Character Inventory**: Review every extracted glyph before exporting.
- **🎉 Celebratory Export**: Smooth .TTF compilation with a celebratory confetti blast 🎊.

---

## 🛠️ The Vision Pipeline

The engine runs a sophisticated **10-stage vision pipeline** to ensure perfect extraction:

1. **🌑 Grayscale**: Color removal for pure luminance analysis.
2. **🌪️ Antialiasing**: Gaussian-based noise reduction for smoother edges.
3. **🌓 Luma Correction**: Dynamic contrast boosting to isolate strokes from background.
4. **🏁 Thresholding**: Intelligent binarization of the image.
5. **🕸️ Edge Topology**: Canny edge detection to identify stroke boundaries.
6. **📍 Geometric Anchors**: Locates the four calibration blocks on your template.
7. **🖼️ Perspective Correction**: Warps the image into a perfect 800x1000 grid.
8. **🗺️ Segment Mapping**: Overlays an 8×8 grid to align character extraction.
9. **✂️ Neural Extraction**: Isolates individual glyph pixels in parallel.
10. **💎 Vectorization**: Converts extracts to SVG paths and compiles the final TTF.

---

## 📝 Character Template

The engine expects a standard **8×8 Character Grid**. Follow the sequence below to ensure your font maps correctly:

| Row   | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 | Col 8 |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** |  `!`  |  `"`  |  `%`  |  `&`  |  `'`  |  `(`  |   -   |   -   |
| **1** |  `)`  |  `+`  |  `,`  |  `-`  |  `.`  |  `/`  |   -   |   -   |
| **2** |  `:`  |  `;`  |  `=`  |  `?`  |  `@`  |  `A`  |  `B`  |  `C`  |
| **3** |  `D`  |  `E`  |  `F`  |  `G`  |  `H`  |  `I`  |  `J`  |  `K`  |
| **4** |  `L`  |  `M`  |  `N`  |  `O`  |  `P`  |  `Q`  |  `R`  |  `S`  |
| **5** |  `T`  |  `U`  |  `V`  |  `W`  |  `X`  |  `Y`  |  `Z`  |  `a`  |
| **6** |  `b`  |  `c`  |  `d`  |  `e`  |  `f`  |  `g`  |  `h`  |  `i`  |
| **7** |  `j`  |  `k`  |  `l`  |  `m`  |  `n`  |  `o`  |  `p`  |  `q`  |

---

## 💻 Technical Stack

- **Framework**: React 19 + TypeScript
- **Styling**: Tailwind CSS 4.0
- **Computer Vision**: OpenCV.js
- **Typography Engine**: opentype.js
- **Vectorization**: Potrace
- **Path Manipulation**: svgpath

---

## 🔨 Development

### Install Dependencies

```bash
npm install
```

### Run Locally

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

---

## 📄 License

This project is open-source and available under the MIT License.

---

Made with ❤️ by [Manish Gun](https://www.linkedin.com/in/manish-gun/)
