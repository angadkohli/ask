/* Shared data + the reusable suitcase SVG used across the site */

// Returns an inline SVG of a suitcase in the given colour.
function suitcaseSVG(color, accent){
  accent = accent || '#ffffff';
  return `
  <svg viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Suitcase">
    <!-- handle -->
    <rect x="78" y="6" width="44" height="34" rx="10" fill="none" stroke="#2b2742" stroke-width="7"/>
    <!-- body -->
    <rect x="34" y="40" width="132" height="160" rx="22" fill="${color}"/>
    <rect x="34" y="40" width="132" height="160" rx="22" fill="url(#sheen)"/>
    <!-- ridges -->
    <rect x="62" y="52" width="6" height="136" rx="3" fill="${accent}" opacity=".25"/>
    <rect x="97" y="52" width="6" height="136" rx="3" fill="${accent}" opacity=".25"/>
    <rect x="132" y="52" width="6" height="136" rx="3" fill="${accent}" opacity=".25"/>
    <!-- corner guards -->
    <circle cx="48" cy="54" r="6" fill="${accent}" opacity=".5"/>
    <circle cx="152" cy="54" r="6" fill="${accent}" opacity=".5"/>
    <!-- wheels -->
    <circle cx="60" cy="206" r="11" fill="#2b2742"/>
    <circle cx="140" cy="206" r="11" fill="#2b2742"/>
    <defs>
      <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#fff" stop-opacity=".28"/>
        <stop offset=".5" stop-color="#fff" stop-opacity="0"/>
      </linearGradient>
    </defs>
  </svg>`;
}

const PRODUCTS = [
  { name:'Yatri Cabin 55', tag:'Bestseller', color:'#e4572e', accent:'#ffd9c9',
    desc:'Lightweight 8-spinner cabin bag. Carry-on approved on IndiGo, Air India & Vistara.',
    price:4499, mrp:6999 },
  { name:'Yatri Check-in 68', tag:'Family fav', color:'#1b998b', accent:'#d6f4ef',
    desc:'Spacious hard-shell check-in for the long Diwali & summer trips. TSA lock.',
    price:6299, mrp:8999 },
  { name:'Yatri Duffel Pro', tag:'Weekender', color:'#3d3a8c', accent:'#dcd9ff',
    desc:'Cabin-friendly wheel duffel for quick weekend getaways to the hills.',
    price:3799, mrp:5499 },
  { name:'Yatri Trolley Trio', tag:'Combo · Save 30%', color:'#f2a900', accent:'#fff1cc',
    desc:'Set of 3 (small + medium + large). Perfect wedding-season gifting set.',
    price:11999, mrp:17999 },
  { name:'Yatri Kids Explorer', tag:'For little ones', color:'#e15a97', accent:'#ffdcec',
    desc:'Ride-on cabin case kids love. Add their name & favourite cartoon print.',
    price:3299, mrp:4499 },
  { name:'Yatri Laptop Backpack', tag:'Daily carry', color:'#2b2742', accent:'#cfcae6',
    desc:'15.6" padded, water-resistant, USB port. Monogram your initials.',
    price:2499, mrp:3499 },
];

function inr(n){ return '₹' + n.toLocaleString('en-IN'); }

// Render the product grid if a #product-grid element exists on the page.
document.addEventListener('DOMContentLoaded', function(){
  const grid = document.getElementById('product-grid');
  if(!grid) return;
  grid.innerHTML = PRODUCTS.map(p => `
    <article class="card">
      <div class="thumb">${suitcaseSVG(p.color, p.accent)}</div>
      <div class="body">
        <span class="badge">${p.tag}</span>
        <h3>${p.name}</h3>
        <p class="desc">${p.desc}</p>
        <div class="row">
          <div class="price">${inr(p.price)} <small>${inr(p.mrp)}</small></div>
          <a class="btn btn-primary" href="studio.html">Personalise</a>
        </div>
      </div>
    </article>`).join('');
});
