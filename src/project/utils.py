import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ANSI color code mapping to RGB
ANSI_COLORS = {
    "32": (0, 255, 0),  # green  → pipes
    "33": (255, 255, 0),  # yellow → bird (@)
    "0": (255, 255, 255),  # reset  → white
}


def parse_ansi(text):
    """Parse a string with ANSI codes into a list of (char, color) tuples.
    Default color is white.
    """
    segments = []
    current_color = (255, 255, 255)  # default white
    # Split on ANSI escape sequences
    pattern = re.compile(r"\x1b\[(\d+)m")
    last_end = 0

    for match in pattern.finditer(text):
        # Text before this escape code
        plain = text[last_end : match.start()]
        for ch in plain:
            segments.append((ch, current_color))
        # Update color
        code = match.group(1)
        current_color = ANSI_COLORS.get(code, (255, 255, 255))
        last_end = match.end()

    # Remaining text after last escape code
    for ch in text[last_end:]:
        segments.append((ch, current_color))

    return segments


def text_render_to_frame(render_str, font_size=16, bg_color=(15, 15, 25), fixed_size=(500, 400)):
    """Convert ANSI text render to an RGB numpy array."""
    lines = render_str.split("\n")
    parsed_lines = [parse_ansi(line) for line in lines]

    char_w = font_size // 2 + 2
    char_h = font_size + 4
    padding = 10

    max_chars = max((len(line) for line in parsed_lines), default=1)
    img_w = max_chars * char_w + 2 * padding
    img_h = len(parsed_lines) * char_h + 2 * padding

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for row, segments in enumerate(parsed_lines):
        x = padding
        y = padding + row * char_h
        for ch, color in segments:
            draw.text((x, y), ch, fill=color, font=font)
            x += char_w

    # Force all frames to identical shape so np.stack works
    img = img.resize(fixed_size, Image.NEAREST)
    return np.array(img)


def record_episode(env, agent, max_steps):
    state, _ = env.reset()
    frames = []
    total_reward = 0

    for step in range(max_steps):
        render_str = env.render()
        frame = text_render_to_frame(render_str)
        frames.append(frame)

        action = agent.step(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    # Capture final frame too
    frames.append(text_render_to_frame(env.render()))
    return frames, total_reward
