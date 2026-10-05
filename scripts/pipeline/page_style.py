"""Shared CSS (Usher default palette, light + dark) for the generated HTML pages."""

CSS = """
:root{--bg:#c2cebc;--surface:#d0dacb;--line:#a8b6a2;--fg:#1c2328;--muted:#4a5660;--accent:#0f5c4f;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12171a;--surface:#192024;--fg:#e3e8eb;--muted:#9aa6ae;--line:#2c363c;--accent:#5cc4ae;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12171a;--surface:#192024;--fg:#e3e8eb;--muted:#9aa6ae;--line:#2c363c;--accent:#5cc4ae;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}
a{color:var(--accent)}p{margin:8px 0}
.sub{color:var(--muted);margin-bottom:18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--line);font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--bg);padding:1px 4px;border-radius:4px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:13px}
button{font:inherit;background:var(--accent);color:var(--bg);border:0;border-radius:6px;padding:5px 12px;cursor:pointer}
button.ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
.theme{position:fixed;top:10px;right:12px}
.muted{color:var(--muted)}
"""

THEME_JS = """
(function(){var b=document.getElementById('themeBtn');if(!b)return;
var k='usher-theme';try{var t=localStorage.getItem(k);if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}
b.onclick=function(){var cur=document.documentElement.getAttribute('data-theme');
var dark=cur?cur==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;
var n=dark?'light':'dark';document.documentElement.setAttribute('data-theme',n);try{localStorage.setItem(k,n);}catch(e){}};})();
"""
