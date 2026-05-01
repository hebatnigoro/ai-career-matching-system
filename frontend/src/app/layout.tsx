import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  variable: "--font-jakarta",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "PathDrift — AI Career Matching",
  description: "Upload CV, pilih target karier, dapatkan analisis kecocokan dan rencana belajar dari AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Inline pre-hydration script: read the user's persisted theme from
  // localStorage and apply it to <html> before React renders. Without
  // this the page paints in light mode and then snaps to dark, which
  // looks broken.
  const themeBootstrap = `
    (function () {
      try {
        var t = localStorage.getItem('pd-theme');
        if (t !== 'dark' && t !== 'light') {
          t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        document.documentElement.setAttribute('data-theme', t);
      } catch (_) {}
    })();
  `;

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${jakarta.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrap }} />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}