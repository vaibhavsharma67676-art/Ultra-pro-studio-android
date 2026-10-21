import fs from 'node:fs';
import path from 'node:path';

const candidates = [
  'android/app/src/main/java/in/vaibhav/ultraprostudio/MainActivity.kt',
  'android/app/src/main/java/in/vaibhav/ultraprostudio/MainActivity.java',
];
const file = candidates.find(p => fs.existsSync(p));
if (!file) {
  console.log('MainActivity not found at expected package path; native back handling will need manual review after Capacitor generation.');
  process.exit(0);
}
let source = fs.readFileSync(file, 'utf8');
if (source.includes('upsNativeBackHandler')) {
  console.log('Native Back handler already installed.');
  process.exit(0);
}
if (file.endsWith('.kt')) {
  source = source.replace(/class MainActivity\s*:\s*BridgeActivity\(\)\s*\{?/, match => `${match.endsWith('{') ? match : `${match} {`}\n  @Suppress("DEPRECATION")\n  override fun onBackPressed() {\n    val webView = bridge?.webView\n    if (webView == null) { super.onBackPressed(); return }\n    webView.evaluateJavascript("(function(){return window.__upsNativeBackHandler ? window.__upsNativeBackHandler() : false})()") { result ->\n      if (result == "true") return@evaluateJavascript\n      if (webView.canGoBack()) webView.goBack()\n      else runOnUiThread {\n        android.app.AlertDialog.Builder(this)\n          .setTitle("Exit Ultra Pro Studio?")\n          .setMessage("Do you want to exit Ultra Pro Studio?")\n          .setNegativeButton("Cancel", null)\n          .setPositiveButton("Exit") { _, _ -> finish() }\n          .show()\n      }\n    }\n  }\n`);
} else {
  source = source.replace(/public class MainActivity extends BridgeActivity\s*\{?/, match => `${match}\n  @Override public void onBackPressed() {\n    android.webkit.WebView webView = getBridge() == null ? null : getBridge().getWebView();\n    if (webView == null) { super.onBackPressed(); return; }\n    webView.evaluateJavascript("(function(){return window.__upsNativeBackHandler ? window.__upsNativeBackHandler() : false})()", result -> {\n      if ("true".equals(result)) return;\n      if (webView.canGoBack()) webView.goBack();\n      else new android.app.AlertDialog.Builder(this).setTitle("Exit Ultra Pro Studio?")\n        .setMessage("Do you want to exit Ultra Pro Studio?")\n        .setNegativeButton("Cancel", null).setPositiveButton("Exit", (d, w) -> finish()).show();\n    });\n  }\n`);
}
fs.writeFileSync(file, source);

// Install the supplied square icon into legacy mipmap slots; adaptive icon XML remains
// template-owned and should be visually reviewed after generation.
try {
  const { execFileSync } = await import('node:child_process');
  const icon = path.resolve('assets/app-icon.png');
  const sizes = { 'mipmap-mdpi': 48, 'mipmap-hdpi': 72, 'mipmap-xhdpi': 96, 'mipmap-xxhdpi': 144, 'mipmap-xxxhdpi': 192 };
  for (const [folder, size] of Object.entries(sizes)) {
    const destDir = path.resolve('android/app/src/main/res', folder);
    if (!fs.existsSync(destDir)) continue;
    for (const name of ['ic_launcher.png', 'ic_launcher_round.png']) {
      const dest = path.join(destDir, name);
      try {
        execFileSync('python3', ['-c', 'from PIL import Image; import sys; Image.open(sys.argv[1]).convert("RGBA").resize((int(sys.argv[2]),int(sys.argv[2]))).save(sys.argv[3])', icon, String(size), dest]);
      } catch (err) {
        console.log(`Could not rasterize launcher icon for ${destDir}; original Capacitor icon remains.`);
      }
    }
  }
} catch (_) {}
console.log(`Installed native Back handler in ${file}`);

