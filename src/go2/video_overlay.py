import pickle

import cv2


def normalize_commands(commands_buffer):
    max_lin_x = max(abs(cmd[0]) for cmd in commands_buffer)
    max_lin_y = max(abs(cmd[1]) for cmd in commands_buffer)
    max_ang_z = max(abs(cmd[2]) for cmd in commands_buffer)
    max_base_height = max(abs(cmd[3]) for cmd in commands_buffer)
    max_jump_height = max(abs(cmd[4]) for cmd in commands_buffer)
    return max_lin_x, max_lin_y, max_ang_z, max_base_height, max_jump_height

def draw_joystick(image, lin_x, lin_y, max_lin_x, max_lin_y, radius=100, x_offset=10, y_offset=10):
    for i in range(radius):
        r = radius - i
        color = (int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)))
        cv2.circle(image, (x_offset + radius, y_offset + radius), r, color, -1)

    joystick_x = int(x_offset + radius + (lin_y / max_lin_y) * radius)
    joystick_y = int(y_offset + radius - (lin_x / max_lin_x) * radius)
    cv2.circle(image, (joystick_x + 2, joystick_y + 2), int(radius * 0.12), (0, 0, 0), -1)

    return image

def draw_target_height_bar(image, base_height, max_base_height, target_height=1.0, x_offset=220, y_offset=10):
    base_height = max(0, base_height) 

    bar_width = 20
    bar_height = 200
    bar_x = x_offset 
    bar_y = y_offset 

    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    current_height_pos = int(bar_y + bar_height - (base_height / max_base_height) * bar_height)
    cv2.rectangle(image, (bar_x, current_height_pos), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)

    target_height_pos = int(bar_y + bar_height - (target_height / target_height) * bar_height)
    cv2.line(image, (bar_x, target_height_pos), (bar_x + bar_width, target_height_pos), (0, 0, 255), 2)

    return image

def draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=10, y_offset=220):
    bar_width = 200
    bar_height = 20
    bar_x = x_offset
    bar_y = y_offset

    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    current_ang_pos = int(bar_x + (ang_z / max_ang_z + 1) / 2 * bar_width)
    cv2.rectangle(image, (bar_x, bar_y), (current_ang_pos, bar_y + bar_height), (0, 255, 0), -1)

    return image

def create_video_with_overlay(images_buffer, commands_buffer, output_video_path, fps=30):
    height, width, _ = images_buffer[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    max_lin_x, max_lin_y, max_ang_z, max_base_height, _ = normalize_commands(commands_buffer)

    for i in range(len(images_buffer)):
        image = images_buffer[i]
        
        lin_x, lin_y, ang_z, base_height, jump_height = commands_buffer[i]

        
        x_offset = images_buffer[0].shape[1] // 2 - 100
        y_offset = images_buffer[0].shape[0]  - 250
        radius = 100
        
        image = draw_joystick(image, lin_x, lin_y, max_lin_x, max_lin_y, radius=radius, x_offset=x_offset, y_offset=y_offset)


        image = draw_target_height_bar(image, base_height, max_base_height, x_offset=x_offset + radius*2 + 10, y_offset=y_offset)

        image = draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=x_offset, y_offset=y_offset + radius*2 + 20)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        out.write(image)

    out.release()

images_buffer = pickle.load(open("images_buffer.pkl", "rb"))
commands_buffer = pickle.load(open("commands_buffer.pkl", "rb"))

OUTPUT_VIDEO_PATH = "output_video.mp4"

create_video_with_overlay(images_buffer, commands_buffer, OUTPUT_VIDEO_PATH, fps=30)
print(f"Video saved to {OUTPUT_VIDEO_PATH}")