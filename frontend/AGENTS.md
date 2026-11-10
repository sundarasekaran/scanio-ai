<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

- Dental screening goes through `analyzeDentalImages` in src/services/prediction.ts: it uses the user's Python model when VITE_PREDICTION_API_URL is set, otherwise the Lovable AI vision server function in src/lib/dental-ai.functions.ts. Why: the user will plug in their own model later without UI changes.
