# Yatri — Personalized Luggage (India) 🧳🇮🇳

A demo website for a **personalized, print-on-demand luggage brand** built for the
Indian market. It includes a homepage, a product collection, and a **live
Personalisation Studio** where customers design their own bag.

![Sample design from the studio](studio-preview.png)

---

## 👀 What's inside

| Page | What it does |
|------|--------------|
| `index.html` | Homepage: hero, features, collection, print-on-demand section, reviews |
| `studio.html` | **The Design Studio** — the main feature |
| `assets/css/style.css` | All the styling (colours, layout) |
| `assets/js/studio.js` | The live design studio logic |
| `assets/js/products.js` | Product list + the suitcase drawing |

### The Design Studio lets a customer:
- Pick the **bag type** (Cabin / Check-in / Duffel / Kids)
- Choose a **base colour** from a palette
- Add their **name or initials** with 3 font styles, colours and sizes
- Add an **Indian-inspired print** (Mandala, Polka, Stripes, Paisley)
- **Upload their own photo/logo/artwork** (print-on-demand)
- See the **₹ price update live** and **Add to cart**
- **Download** their design as an image

---

## ▶️ How to see it on your computer (no coding)

**Easiest way:** just double-click `index.html` — it opens in your web browser.

**Slightly better way** (so uploads & fonts work perfectly), in a terminal:

```bash
# from inside this folder
python3 -m http.server 8080
```
Then open **http://localhost:8080** in your browser.

---

## 🚀 How to put it online (free)

This is a plain website (HTML/CSS/JS), so hosting is very easy:

1. Go to **netlify.com** (or **vercel.com**) and make a free account.
2. Drag-and-drop this whole folder onto their "deploy" page.
3. You'll get a live link like `your-brand.netlify.app`.
4. (Optional) Buy a domain like `yatri.in` and connect it.

---

## 🛠️ This is a starting point — what's NOT included yet

This demo focuses on the **look and the design studio**. A real shop would
also need (we can add these step by step):

- Real **checkout & payments** (Razorpay / UPI for India)
- A **database** to store orders and designs
- **Login / accounts**
- An **admin panel** to manage products and orders
- Connecting designs to an actual **print/fulfilment** partner

> Built as a first version. Ask Claude to extend any part of it.
