# Ultra Pro Studio — UI refresh + Capacitor Android wrapper

## What changed

- `indexx.html`: adds a persistent Dark/Light theme switch, responsive layout refinements, and a native-Back callback while preserving the existing page's tool markup and application scripts.
- Footer now says **Created by Vaibhav**.
- `Dockerfile`: added as a copy of the supplied `Dockerfile.txt` because Hugging Face Docker Spaces conventionally look for a root file named `Dockerfile`. `Dockerfile.txt` remains unchanged.
- `capacitor.config.ts`, `package.json`, `www/index.html`: Capacitor wrapper configuration. The Android app loads the real HTTPS Hugging Face site, because Flask/Python audio processing remains server-side.
- `.github/workflows/android.yml`: cloud workflow to build a debug APK or unsigned release AAB.
- `scripts/prepare-android.mjs`: adds native Android Back confirmation after Capacitor generates the Android project.
- `assets/app-icon.svg` and `assets/app-icon.png`: source app icon artwork.
- `original-backup/`: exact copies of key original files from the uploaded ZIP.

## Important: set the real Hugging Face URL

The uploaded ZIP does not reliably identify the production Space URL. Do not guess it.

In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables → New repository variable** and set:

- Name: `APP_SERVER_URL`
- Value: your live Space URL, e.g. `https://YOUR-SPACE-NAME.hf.space`

Use your actual URL, not the example. The app requires HTTPS; cleartext HTTP is rejected.

## Build an APK from an Android phone

1. Upload/extract this project into a GitHub repository (GitHub mobile app or github.com in your phone browser).
2. Set the `APP_SERVER_URL` repository Actions variable as described above.
3. Open the repository's **Actions** tab.
4. Select **Build Ultra Pro Studio Android** → **Run workflow**.
5. Choose `debug-apk` and run it.
6. Open the completed workflow run, scroll to **Artifacts**, and download `ultra-pro-studio-debug-apk`.
7. Extract the artifact ZIP on your phone, tap `app-debug.apk`, and approve installation from that source if Android asks.

A debug APK is for testing and is not a Play Store release.

## Build locally (if Node 20+, Java 21, Android SDK are installed)

```bash
npm install
export APP_SERVER_URL="https://YOUR-SPACE-NAME.hf.space"
npx cap add android
node scripts/prepare-android.mjs
npx cap sync android
cd android
./gradlew assembleDebug
```

APK output: `android/app/build/outputs/apk/debug/app-debug.apk`.

For an unsigned release bundle, run `./gradlew bundleRelease`; output is normally `android/app/build/outputs/bundle/release/app-release.aab`. Sign the AAB using a secure release keystore before Play Console upload. Never commit a keystore or its password to a public repository.

## Architecture and limitations

- The Capacitor app loads the production HTTPS site; it does **not** bundle or run Flask/Python inside Android.
- Audio uploads, server-side processing, and output URLs remain dependent on the Hugging Face Space being online and allowing the existing workflow. This package has not been tested against your live Space because its production URL and credentials/session behavior were not included.
- The debug APK has **not** been built in this environment. GitHub Actions is configured to build it after `APP_SERVER_URL` is set.
- The native Back handler is added to the generated Capacitor MainActivity by the preparation script. Review its build output in the Actions run; Capacitor/Android template changes can require a small compatibility adjustment.
- The app icon artwork is included as a source asset; launcher/splash resource installation and final visual validation on a real Android device remain to be verified.
- No claim is made that audio upload, processing, preview, download, or Play Store compliance has been device-tested.
- A remote website wrapper may face Play review scrutiny if it provides little app-specific value. Confirm current Play policy, provide a privacy policy that accurately describes uploaded audio and server retention, use signed AAB, and complete Play Console declarations/testing requirements before publication.

## Source preservation / change list

Modified:
- `indexx.html` — theme, responsive CSS, footer, Back callback.
- `package.json` — replaced unrelated Vite/Hono manifest with Capacitor dependencies.
- `capacitor.config.ts` — added.
- `.github/workflows/android.yml` — added.
- `scripts/prepare-android.mjs` — added.
- `www/index.html` — fallback page for Capacitor setup.
- `Dockerfile` — added from supplied `Dockerfile.txt`.

Added assets/docs:
- `assets/app-icon.svg`, `assets/app-icon.png`
- `README_ANDROID.md`
- `original-backup/*`

Untouched from the source ZIP:
- `server.py` (all Flask routes and audio processing)
- `requirements.txt`
- `Dockerfile.txt`
- `README.md`, `README_DEPLOY.txt`
- `MANIFEST.txt`, `gitattributes.txt`, `env.example.txt`, `gitignore.txt`
- `packages.txt`, `pnpm-workspace.yaml`, `tsconfig.json`, `vite.config.ts`, `watermark_video (2).py`

The original `package-lock.json` is preserved under `original-backup/` because it belongs to the unrelated original Node/Vite setup and does not match the new Capacitor package manifest.
