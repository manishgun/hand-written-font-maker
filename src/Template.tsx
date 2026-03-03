// A4Template.tsx
export default function A4Template() {
  const rows = 11;
  const cols = 9;

  const characters = [
    ..."abcdefghijklmnopqrstuvwxyz".split(""),
    ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""),
    ..."0123456789".split(""),
    ..."!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".split(""),
  ];

  return (
    <div className="bg-gray-200 flex justify-center py-10 print:bg-white print:py-0">
      <div
        className="relative bg-white"
        style={{
          width: "210mm",
          height: "297mm",
          padding: "20mm 6mm",
        }}>
        <div className="text-2xl font-semibold top-20 absolute left-1/2 -translate-x-1/2 text-black/70">HAND-WRITTEN FONT MAKER</div>
        <div className="text-xl font-semibold bottom-6 absolute left-1/2 -translate-x-1/2 text-black/20">Made By Manish Gun</div>
        {/* Corner Markers */}
        <Marker position="top-left" />
        <Marker position="top-right" />
        <Marker position="bottom-left" />
        <Marker position="bottom-right" />

        {/* Grid */}
        <div className="grid grid-cols-9 grid-rows-11 w-full h-full">
          {Array.from({ length: rows * cols }).map((_, i) => (
            <div key={i} className="border border-black/10">
              <div className="w-full h-[4mm] border-b border-black/10 text-[10px] text-black/70 pl-1">{characters[i] ?? ""}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Marker({ position }: { position: string }) {
  //   const base = "absolute bg-black";

  const positionMap: Record<string, React.CSSProperties> = {
    "top-left": { top: "6mm", left: "6mm" },
    "top-right": { top: "6mm", right: "6mm" },
    "bottom-left": { bottom: "6mm", left: "6mm" },
    "bottom-right": { bottom: "6mm", right: "6mm" },
  };

  return (
    <div
      className={"absolute bg-black w-[12mm] h-[12mm] flex items-center justify-center"}
      style={{
        ...positionMap[position],
      }}>
      <div className="w-[8mm] h-[8mm] bg-white flex justify-center items-center">
        <div className="w-[4mm] h-[4mm] bg-black" />
      </div>
    </div>
  );
}
