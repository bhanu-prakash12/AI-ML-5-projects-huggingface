import gradio as gr

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")
CSS = """
.hub-hero{padding:28px;border-radius:16px;background:linear-gradient(135deg,#1565C0,#6A1B9A 55%,#C62828);color:#fff}
.hub-hero h1{margin:0 0 6px;color:#fff;font-size:2em}.hub-hero p{margin:4px 0;color:#f3f3ff}
.hub-card{border-radius:14px;padding:16px;border:1px solid #d0d7de;background:var(--background-fill-primary);min-height:210px}
.hub-card h3{margin:6px 0}.hub-card small{opacity:.8}
.hub-pill{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-size:.8em;font-weight:600}
.hub-kind{background:#455A64!important}
.hub-banner{padding:12px 16px;border-radius:10px;background:var(--background-fill-secondary);margin-bottom:10px}
.hub-pitch{margin-top:4px}.hub-src{margin-top:4px;font-size:.85em;opacity:.75}
.hub-steps li{margin:6px 0}
"""
