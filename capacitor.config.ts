import type { CapacitorConfig } from '@capacitor/cli';

const remoteUrl = process.env.APP_SERVER_URL?.trim();
if (!remoteUrl) {
  console.warn('APP_SERVER_URL is not set. Set it to the production HTTPS Hugging Face Space URL before building the Android app.');
}
if (remoteUrl && !remoteUrl.startsWith('https://')) {
  throw new Error('APP_SERVER_URL must use HTTPS. Cleartext HTTP is intentionally disabled.');
}

const config: CapacitorConfig = {
  appId: 'in.vaibhav.ultraprostudio',
  appName: 'Ultra Pro Studio',
  webDir: 'www',
  server: {
    ...(remoteUrl ? { url: remoteUrl } : {}),
    cleartext: false,
    androidScheme: 'https',
    ...(remoteUrl ? { allowNavigation: [new URL(remoteUrl).hostname] } : {}),
  },
  android: {
    allowMixedContent: false,
    captureInput: true,
  },
  plugins: {
    App: { },
    StatusBar: {
      style: 'DARK',
      backgroundColor: '#080b14',
    },
  },
};

export default config;
