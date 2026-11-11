import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, ImagePlus, Loader2, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, Disclaimer } from "@/components/shared";
import { analyzeDentalImages, VIEWS, wakeBackend, type ViewKey } from "@/services/prediction";
import { saveScreening, useUser } from "@/services/store";

export const Route = createFileRoute("/upload")({
  head: () => ({ meta: [{ title: "Start Screening — Scanio AI" }, { name: "description", content: "Upload upper arch, lower arch and front view photos for AI-assisted screening." }, { property: "og:title", content: "Start Screening — Scanio AI" }, { property: "og:description", content: "Three photos, one AI-assisted dental screening." }] }),
  component: UploadPage,
});

const TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const EXT_RE = /\.(jpe?g|png|webp)$/i;
const MAX_UPLOAD_BYTES = 8_000_000;

// Downscale so three photos fit in browser storage.
function toDataUrl(f: File) {
  return new Promise<string>((res, rej) => {
    const objectUrl = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      try {
        const s = Math.min(1, 900 / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(img.width * s));
        c.height = Math.max(1, Math.round(img.height * s));
        c.getContext("2d")!.drawImage(img, 0, 0, c.width, c.height);
        res(c.toDataURL("image/jpeg", 0.8));
      } catch (err) {
        rej(err);
      } finally {
        URL.revokeObjectURL(objectUrl);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      rej(new Error("Could not decode the selected image file."));
    };
    img.src = objectUrl;
  });
}

type Slot = { file: File; preview: string };

function UploadSlot({ label, hint, slot, disabled, onPick, onClear }: { label: string; hint: string; slot?: Slot | undefined; disabled: boolean; onPick: (f?: File) => void; onClear: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <Card className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between"><h3 className="font-semibold">{label}</h3>{slot && <CheckCircle2 className="h-5 w-5 text-success" />}</div>
      {slot ? (
        <div className="relative overflow-hidden rounded-xl bg-muted">
          <img src={slot.preview} alt={`${label} preview`} className="aspect-[4/3] w-full object-cover" />
          {!disabled && <button onClick={onClear} className="absolute right-2 top-2 rounded-full bg-background p-1.5 shadow-card" aria-label={`Remove ${label}`}><X className="h-4 w-4" /></button>}
        </div>
      ) : (
        <button type="button" disabled={disabled} onClick={() => input.current?.click()}
          onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onPick(e.dataTransfer.files[0]); }}
          className="flex aspect-[4/3] flex-col items-center justify-center rounded-xl border-2 border-dashed border-border px-4 text-center transition hover:border-primary hover:bg-accent/50">
          <span className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-primary text-primary-foreground"><ImagePlus className="h-5 w-5" /></span>
          <p className="mt-3 text-sm font-medium">Upload {label.toLowerCase()}</p>
          <p className="text-xs text-muted-foreground">JPG, JPEG, PNG or WEBP</p>
        </button>
      )}
      <p className="text-xs text-muted-foreground">{hint}</p>
      {slot && !disabled && <Button variant="outline" size="sm" onClick={() => input.current?.click()}>Change photo</Button>}
      <input ref={input} type="file" aria-label={`${label} file`} accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => { onPick(e.target.files?.[0]); e.target.value = ""; }} />
    </Card>
  );
}

function UploadPage() {
  const { user, ready } = useUser();
  const nav = useNavigate();
  const [slots, setSlots] = useState<Partial<Record<ViewKey, Slot>>>({});
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const done = VIEWS.filter((v) => slots[v.key]).length;

  useEffect(() => {
    wakeBackend();
  }, []);

  const pick = async (key: ViewKey, f?: File) => {
    if (!f) return;
    const mime = f.type.toLowerCase();
    const validType = mime ? TYPES.includes(mime) : EXT_RE.test(f.name);
    if (!validType) {
      toast.error("Please upload a JPG, JPEG, PNG or WEBP image");
      return;
    }
    if (f.size === 0) {
      toast.error("Selected image file is empty");
      return;
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      toast.error("Image must be under 8 MB");
      return;
    }
    try {
      const preview = await toDataUrl(f);
      setErrorMsg(null);
      setSlots((s) => ({ ...s, [key]: { file: f, preview } }));
    } catch {
      toast.error("Could not read this image file. Please choose a valid JPG, PNG, or WEBP photo.");
    }
  };

  const analyze = async () => {
    if (!user || done < 3) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      const files = { upper: slots.upper!.file, lower: slots.lower!.file, front: slots.front!.file };
      const images = { upper: slots.upper!.preview, lower: slots.lower!.preview, front: slots.front!.preview };
      const result = await analyzeDentalImages(files, images);
      const s = saveScreening({ ...result, images: { upper: slots.upper!.preview, lower: slots.lower!.preview, front: slots.front!.preview }, userEmail: user.email });
      nav({ to: "/results/$id", params: { id: s.id } });
    } catch (e) {
      const msg = e instanceof Error && e.message ? e.message : "Dental screening failed. Please try again.";
      setErrorMsg(msg);
      toast.error(msg);
      setBusy(false);
    }
  };

  if (ready && !user) return (
    <div className="mx-auto max-w-md px-4 py-20 text-center">
      <h1 className="text-2xl font-bold">Log in to start screening</h1>
      <p className="mt-2 text-muted-foreground">An account lets us save your results to your dashboard.</p>
      <div className="mt-6 flex justify-center gap-3"><Button asChild><Link to="/login">Log in</Link></Button><Button variant="outline" asChild><Link to="/register">Sign up</Link></Button></div>
    </div>
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-12">
      <div><h1 className="text-3xl font-bold">Dental screening</h1><p className="mt-1 text-muted-foreground">Upload 3 clear, well-lit photos: upper arch, lower arch and front view.</p></div>
      <div className="grid gap-4 md:grid-cols-3">
        {VIEWS.map((v) => (
          <UploadSlot key={v.key} label={v.label} hint={v.hint} slot={slots[v.key]} disabled={busy}
            onPick={(f) => pick(v.key, f)} onClear={() => setSlots((s) => { const n = { ...s }; delete n[v.key]; return n; })} />
        ))}
      </div>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-muted-foreground">{done} of 3 photos added</p>
        <Button size="lg" disabled={busy || done < 3} onClick={analyze}>{busy ? <><Loader2 className="animate-spin" /> Analyzing…</> : "Analyze photos"}</Button>
      </div>
      {errorMsg && (
        <div role="alert" className="flex items-start gap-2.5 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <p>{errorMsg}</p>
        </div>
      )}
      <Disclaimer />
    </div>
  );
}
