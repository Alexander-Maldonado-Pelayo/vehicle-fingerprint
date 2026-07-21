# Publishing this project to GitHub

Step-by-step guide to get this repo online. The repo is already initialized and has
its first commit — you just need to authenticate and push.

---

## Option 1 — GitHub CLI (recommended, no manual token)

The easiest path. `gh auth login` handles authentication in your browser, so you never
create or paste a token by hand.

```bash
cd C:\Users\maldo\car_detect

# one-time login (choose: GitHub.com -> HTTPS -> Login with a web browser)
gh auth login

# create the repo AND push in one command
gh repo create vehicle-fingerprint --public --source=. --push
```

Done — it prints the URL of your new repo.

> Don't have `gh`? Install it from https://cli.github.com/ (or `winget install GitHub.cli`).

---

## Option 2 — Manual, with a Personal Access Token (PAT)

GitHub no longer accepts your account password for `git push`; you use a token instead.

### Create the token
1. Go to **https://github.com/settings/tokens**
   (GitHub → avatar → **Settings** → **Developer settings** → **Personal access tokens**)
2. **Fine-grained tokens** → **Generate new token**
3. Configure:
   - **Name:** `vehicle-fingerprint-push`
   - **Expiration:** 90 days
   - **Repository access:** Only select repositories (pick this repo once it exists) or All
   - **Permissions → Repository → Contents:** *Read and write*  ← the one push needs
4. **Generate token** → **copy it now** (shown only once).

### Create the repo and push
1. On github.com, create a new **empty** repo named `vehicle-fingerprint`
   (no README/license — this project already has them).
2. Then:

```bash
cd C:\Users\maldo\car_detect
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vehicle-fingerprint.git
git push -u origin main
```

3. When prompted:
   - **Username:** your GitHub username
   - **Password:** paste the **token** (not your account password)

---

## Security notes

- A token is a credential — treat it like a password.
- **Never** commit it, paste it into a file, or share it in chat.
- If it leaks, revoke it immediately at https://github.com/settings/tokens.
- Prefer Option 1 (`gh auth login`) — it stores credentials securely and avoids
  handling a raw token at all.

---

## What gets uploaded

`.gitignore` already excludes the heavy/generated files. Only source is pushed:

**Pushed:** all `.py` files, `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`,
this `PUBLISHING.md`.

**Excluded:** `yolov8n.pt`, `fingerprint.pth`, sample videos (`*.mp4`/`*.avi`),
`output.*`, `vehicles.csv`, `.venv/`, `.vscode/`, `__pycache__/`.

---

## After pushing — nice touches for a portfolio repo

- Add repo **topics** on GitHub: `computer-vision`, `yolov8`, `object-tracking`,
  `pytorch`, `opencv`, `vehicle-detection`.
- Add a short **description** and, if you have one, a demo GIF/screenshot in the README.
- Set the **About** section's website to a demo video if you host one.
