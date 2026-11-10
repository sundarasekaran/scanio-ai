import { createFileRoute, Link } from "@tanstack/react-router";
import { Camera, FileText, ShieldCheck, Sparkles, Upload, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/shared";
import hero from "@/assets/hero-dental.jpg";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Scanio AI — AI-Assisted Dental Screening" },
      { name: "description", content: "Upload a photo of your teeth and get an instant AI-assisted dental screening with recommendations." },
      { property: "og:title", content: "Scanio AI — AI-Assisted Dental Screening" },
      { property: "og:description", content: "Instant AI-assisted dental screening from a simple photo." },
    ],
  }),
  component: Home,
});

const features = [
  { icon: Camera, title: "Snap a photo", text: "Use your phone camera to capture a clear image of your teeth." },
  { icon: Sparkles, title: "AI screening", text: "Our model highlights possible concerns like plaque, cavities and gum issues." },
  { icon: FileText, title: "Get your report", text: "Download a report with confidence scores and next-step recommendations." },
];

function Home() {
  return (
    <>
      <section className="bg-gradient-soft">
        <div className="mx-auto grid max-w-6xl items-center gap-10 px-4 py-16 md:grid-cols-2 md:py-24">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full bg-accent px-3 py-1 text-xs font-semibold text-accent-foreground"><ShieldCheck className="h-4 w-4" /> AI-assisted dental screening</span>
            <h1 className="mt-5 text-4xl font-bold leading-tight md:text-6xl">Check your smile <span className="bg-gradient-primary bg-clip-text text-transparent">in seconds</span></h1>
            <p className="mt-5 max-w-lg text-lg text-muted-foreground">Upload a photo of your teeth and Scanio AI screens for common dental concerns — so you know when it's time to see a dentist.</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Button size="lg" asChild><Link to="/upload"><Upload /> Start Screening</Link></Button>
              <Button size="lg" variant="outline" asChild><Link to="/register">Create free account</Link></Button>
            </div>
            <div className="mt-10 flex gap-8">
              {[["3 steps", "Simple process"], ["< 1 min", "Screening time"], ["24/7", "Available"]].map(([a, b]) => (
                <div key={a}><p className="font-display text-2xl font-bold text-primary">{a}</p><p className="text-xs text-muted-foreground">{b}</p></div>
              ))}
            </div>
          </div>
          <div className="relative">
            <img src={hero} alt="Patient scanning teeth with a smartphone at a dental clinic" width={1280} height={1024} className="rounded-3xl shadow-card" />
            <Card className="absolute -bottom-6 left-4 flex items-center gap-3 p-4">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-accent text-primary"><Activity /></span>
              <div><p className="text-xs text-muted-foreground">Oral health score</p><p className="font-display text-lg font-bold">86 / 100</p></div>
            </Card>
          </div>
        </div>
      </section>
      <section className="mx-auto max-w-6xl px-4 py-20">
        <h2 className="text-center text-3xl font-bold">How it works</h2>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {features.map((f, i) => (
            <Card key={f.title}>
              <span className="grid h-12 w-12 place-items-center rounded-xl bg-gradient-primary text-primary-foreground"><f.icon /></span>
              <p className="mt-4 text-xs font-semibold text-primary">STEP {i + 1}</p>
              <h3 className="mt-1 text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.text}</p>
            </Card>
          ))}
        </div>
      </section>
      <section className="mx-auto max-w-6xl px-4 pb-20">
        <div className="rounded-3xl bg-gradient-primary p-10 text-center text-primary-foreground md:p-14">
          <h2 className="text-3xl font-bold">Ready for your free screening?</h2>
          <p className="mx-auto mt-3 max-w-md opacity-90">It takes less than a minute. Results are for guidance only and don't replace a dentist visit.</p>
          <Button size="lg" variant="secondary" className="mt-6" asChild><Link to="/upload">Start Screening</Link></Button>
        </div>
      </section>
    </>
  );
}
