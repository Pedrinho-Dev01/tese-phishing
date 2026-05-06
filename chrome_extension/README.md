# Gone Phishing — Chrome Extension

Scan the email currently open in **Gmail** or **Outlook Web** and get instant spam + emotional manipulation analysis from the Gone Phishing ensemble API.

## Installation (Developer Mode)

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle, top-right)
3. Click **Load unpacked** and select this folder (`gone-phishing-extension/`)
4. The 🎣 icon will appear in your toolbar

## Usage

1. Open Gmail (`mail.google.com`) or Outlook Web (`outlook.live.com`, `outlook.office.com`)
2. Click on any email to open it
3. Click the Gone Phishing toolbar icon
4. Hit **Scan email**

The extension extracts the email's subject, sender, and body, then calls:
- `POST /predict` — spam/ham ensemble classification (RoBERTa-Large + ELECTRA-Large)
- `POST /predict/emotion` — multi-label emotion detection (8 consolidated classes)

Results are shown in a popup matching the main web app's dark aesthetic.

## Supported Clients

| Client | URL |
|--------|-----|
| Gmail  | `mail.google.com` |
| Outlook Live | `outlook.live.com` |
| Outlook Office | `outlook.office.com` / `outlook.office365.com` |

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Extension manifest (MV3) |
| `content.js` | Injected into mail pages; extracts email text |
| `background.js` | Service worker; makes API requests |
| `popup.html/css/js` | Toolbar popup UI |
| `icons/` | Extension icons (16 / 48 / 128 px) |

## API

Points to `https://dpedrinho01-api-host.hf.space` (the hosted Hugging Face Space).  
No API key required.
