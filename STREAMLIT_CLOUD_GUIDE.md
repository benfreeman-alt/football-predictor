# 🌐 Deploy to Streamlit Cloud - Step by Step

Access your dashboard from anywhere in the world!

---

## 📋 PREREQUISITES:

- [ ] GitHub account
- [ ] Git installed on PC
- [ ] Your dashboard working locally
- [ ] 30 minutes of time

---

## 🚀 STEP-BY-STEP GUIDE:

### **STEP 1: Install Files** (2 minutes)

```bash
cd C:\Users\User\Downloads

# Move all deployment files
move requirements.txt C:\Users\User\Desktop\prediction-markets\
move .gitignore C:\Users\User\Desktop\prediction-markets\
move README.md C:\Users\User\Desktop\prediction-markets\
move .streamlit C:\Users\User\Desktop\prediction-markets\

cd C:\Users\User\Desktop\prediction-markets
```

---

### **STEP 2: Initialize Git** (3 minutes)

```bash
# Configure Git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - Football prediction dashboard"
```

**Expected output:**
```
Initialized empty Git repository
[main (root-commit) abc1234] Initial commit
 50 files changed, 5000 insertions(+)
```

---

### **STEP 3: Create GitHub Repository** (5 minutes)

1. **Go to:** https://github.com/new

2. **Fill in:**
   - Repository name: `football-predictor`
   - Description: `74.4% accuracy football betting dashboard`
   - ⚪ Public (must be public for free Streamlit hosting)
   - ☐ Don't initialize with README (we already have one)

3. **Click:** "Create repository"

4. **Copy the URL:** 
   ```
   https://github.com/YOUR_USERNAME/football-predictor.git
   ```

---

### **STEP 4: Push Code to GitHub** (3 minutes)

```bash
# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/football-predictor.git

# Push code
git branch -M main
git push -u origin main
```

**If asked for credentials:**
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)

**To create token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Select "repo" scope → Generate
3. Copy token and use as password

**Expected output:**
```
Enumerating objects: 100, done.
Writing objects: 100% (100/100), 50KB | 5MB/s, done.
Total 100 (delta 20), reused 0 (delta 0)
To https://github.com/YOUR_USERNAME/football-predictor.git
 * [new branch]      main -> main
```

---

### **STEP 5: Verify on GitHub** (1 minute)

1. Go to: `https://github.com/YOUR_USERNAME/football-predictor`
2. You should see:
   - ✅ dashboard.py
   - ✅ universal_framework/
   - ✅ data/
   - ✅ requirements.txt
   - ✅ README.md

---

### **STEP 6: Deploy on Streamlit Cloud** (10 minutes)

1. **Go to:** https://share.streamlit.io/signup

2. **Sign up with GitHub:**
   - Click "Continue with GitHub"
   - Authorize Streamlit Cloud
   - Accept permissions

3. **Create new app:**
   - Click "New app" button
   - **Repository:** `YOUR_USERNAME/football-predictor`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
   - Click "Deploy!"

4. **Wait for deployment:**
   ```
   Installing requirements...
   ✅ streamlit
   ✅ pandas
   ✅ numpy
   ... etc
   
   Starting app...
   ✅ App is live!
   ```

---

### **STEP 7: Add API Keys (Secrets)** (5 minutes)

**Your app will error without API keys! Add them now:**

1. **In Streamlit Cloud dashboard:**
   - Click your app
   - Click "Settings" (⚙️)
   - Click "Secrets"

2. **Add secrets** (copy/paste this format):
   ```toml
   FOOTBALL_DATA_TOKEN = "your_actual_token_here"
   ODDS_API_KEY = "your_actual_odds_key_here"
   ```

3. **Get your tokens:**
   - FOOTBALL_DATA_TOKEN: From football-data.org dashboard
   - ODDS_API_KEY: From the-odds-api.com dashboard

4. **Click "Save"**

5. **App will restart automatically**

---

### **STEP 8: Test Your App!** (2 minutes)

**Your URL:**
```
https://YOUR_USERNAME-football-predictor.streamlit.app
```

**Test from:**
- ✅ Your PC browser
- ✅ Your phone browser
- ✅ Different WiFi network
- ✅ Share with friends!

**Check:**
- ✅ Fixtures load
- ✅ Predictions show
- ✅ Bet tracking works
- ✅ Odds display

---

## 🎉 SUCCESS! YOU'RE LIVE!

### **Your Dashboard is Now:**
- ✅ Accessible from anywhere
- ✅ On any device
- ✅ Always up-to-date
- ✅ Free forever!

### **Share Your URL:**
```
https://YOUR_USERNAME-football-predictor.streamlit.app
```

---

## 🔄 UPDATING YOUR APP:

**When you make changes:**

```bash
cd C:\Users\User\Desktop\prediction-markets

# Make your changes to dashboard.py or other files

# Commit and push
git add .
git commit -m "Updated dashboard"
git push

# Streamlit Cloud auto-deploys! ✅
```

**App updates automatically in 1-2 minutes!**

---

## 🐛 TROUBLESHOOTING:

### **Problem: "Module not found"**

**Solution:** Add missing package to `requirements.txt`

```bash
# Edit requirements.txt, add:
missing-package>=1.0.0

# Push update
git add requirements.txt
git commit -m "Added missing package"
git push
```

---

### **Problem: "Secrets not found"**

**Solution:** Check secrets in Streamlit Cloud settings

1. Settings → Secrets
2. Verify format:
   ```toml
   KEY = "value"
   ```
3. No extra spaces or quotes

---

### **Problem: "App keeps restarting"**

**Solution:** Check logs

1. Streamlit Cloud → Your app
2. Click "Logs"
3. See error message
4. Fix issue in code
5. Push update

---

## 📱 MOBILE APP FEEL:

### **Add to Home Screen:**

**iPhone:**
1. Open your URL in Safari
2. Tap share button (⬆️)
3. "Add to Home Screen"
4. Now it's like a native app!

**Android:**
1. Open your URL in Chrome
2. Menu (⋮) → "Add to Home Screen"
3. Icon appears on home screen
4. Opens like regular app!

---

## 💡 PRO TIPS:

### **Custom Domain (Optional):**

Want `betting.yourdomain.com` instead of `streamlit.app`?

1. Buy domain (namecheap, etc)
2. Streamlit Cloud → Settings → Custom domain
3. Add CNAME record
4. Done!

---

### **Private App (Paid):**

Want password protection?

1. Upgrade to Streamlit Cloud Pro ($20/month)
2. Add authentication
3. Only you can access

---

### **Multiple Environments:**

Want test + production?

1. Create branch: `git checkout -b dev`
2. Deploy `dev` branch separately
3. Test on dev URL
4. Merge to main when ready

---

## 🎯 WHAT YOU'VE ACHIEVED:

✅ Professional betting dashboard  
✅ Accessible from anywhere  
✅ Works on all devices  
✅ Auto-updates  
✅ Free hosting  
✅ Secure API keys  

**This is better than most paid services!** 🚀

---

## 📊 ANALYTICS (Optional):

Track usage with Streamlit Cloud analytics:

1. Your app → Analytics
2. See:
   - Daily active users
   - Session duration
   - Most used features
   - Error rates

---

## 🔒 SECURITY NOTES:

**Your API keys are safe:**
- ✅ Stored securely on Streamlit Cloud
- ✅ Never visible in code
- ✅ Never in GitHub
- ✅ Encrypted in transit

**Your code is public but:**
- ✅ No sensitive data in code
- ✅ All secrets in Streamlit settings
- ✅ Can make private repo (paid GitHub)

---

## ✅ FINAL CHECKLIST:

- [ ] Git installed and configured
- [ ] Code pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] API keys added to secrets
- [ ] App loads without errors
- [ ] Tested on phone
- [ ] Bookmarked URL
- [ ] Added to home screen

**Done? Congratulations! 🎉**

**Your betting system is now live and accessible from anywhere in the world!**

---

## 📞 NEED HELP?

**Stuck? Show me:**
1. The step you're on
2. Any error messages
3. Screenshots if needed

**I'll help you get it working!** 💪
