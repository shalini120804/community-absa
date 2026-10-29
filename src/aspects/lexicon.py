ASPECTS = {
    "price":        ["price", "cost", "value", "money", "deal", "bargain"],
    "quality":      ["quality", "material", "build", "construction", "plastic", "rubber", "leather"],
    "durability":   ["durability", "lifespan", "sturdiness"],
    "fit":          ["fit", "size", "cutout", "opening", "hole"],
    "battery":      ["battery", "power", "juice", "mah"],
    "charging":     ["charger", "charge", "charging", "cable", "cord", "adapter", "port", "usb"],
    "screen":       ["screen", "display", "protector", "glass", "touchscreen", "resolution"],
    "camera":       ["camera", "photo", "picture", "lens", "video"],
    "sound":        ["sound", "speaker", "volume", "audio", "mic", "microphone", "bass", "call"],
    "design":       ["design", "look", "color", "style", "appearance", "feel", "grip"],
    "installation": ["installation", "install", "setup", "instruction", "application"],
    "performance":  ["performance", "speed", "signal", "connection", "bluetooth", "reception", "wifi", "lag"],
    "service":      ["service", "shipping", "delivery", "seller", "packaging", "support"],
}
LEMMA2ASPECT = {w: a for a, ws in ASPECTS.items() for w in ws}