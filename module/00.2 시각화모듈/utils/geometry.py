def calculate_lines(center, pitch, max_die, min_die):
    lines = []
    current = center
    for _ in range(max_die + 2):
        lines.append(current)
        current += pitch
    current = center
    for _ in range(abs(min_die) + 1):
        current -= pitch
        lines.append(current)
    return lines
