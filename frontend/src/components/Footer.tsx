import { Logo } from "./Navbar";

export function Footer() {
  return (
    <footer className="border-t border-border bg-muted">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 md:flex-row md:items-center md:justify-between">
        <Logo />
        <p className="max-w-xl text-xs text-muted-foreground">Scanio AI provides AI-assisted screening for informational purposes only. It is not a medical diagnosis. Always consult a licensed dentist.</p>
      </div>
    </footer>
  );
}
