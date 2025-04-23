import base64
import os
from io import BytesIO
from tkinter import Image

from PIL import Image,ImageDraw,ImageFont
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template
from flask import redirect, url_for, flash
from flask import request
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

matplotlib.rcParams['animation.embed_limit'] = 2**128


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'  # Define your upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_charge_height(rock_density, explosive_density, hole_length):
    return (rock_density / explosive_density)* hole_length

def calculate_ppv(distance,charge_per_hole, k, e):
    scaled_distance = distance / (charge_per_hole**0.5)
    return k * (scaled_distance ** -e)

def generate_blasting_pattern(pattern_type, num_holes, burden, spacing, num_rows):
    positions = []
    cols = num_holes // num_rows
    for i in range(num_rows):
        for j in range(cols):
            if pattern_type == 'Square':
                positions.append((j * spacing, i * spacing))
            elif pattern_type == 'Staggered':
                x_offset = j * spacing + (spacing / 2 if i % 2 == 1 else 0)
                positions.append((x_offset, i * burden))
            else:
                raise ValueError("Unsupported Pattern type. Use 'square' or 'staggered'.")
    return positions


def plot_blasting_pattern(positions, burden, spacing, num_rows, connection_type, row_delay=None, diagonal_delay=None,
                          pattern_type=None,free_faces=[]):
    x, y = zip(*positions)
    fig,ax = plt.subplots(figsize=(12, 6))
    #ax.set_title('Blasting Pattern')
    ax.set_xticks([])
    ax.set_yticks([])
    scatter= ax.scatter(x, y, c='blue', s=100, edgecolors='black')

    delays = [None] * len(positions)
    last_row_start = (num_rows - 1) * (len(positions) // num_rows)
    delays[last_row_start] = 0

    if row_delay is not None:
        for i in range(last_row_start + 1, len(positions)):
            delays[i] = delays[i-1] + row_delay

        for row in range(num_rows - 2, -1, -1):
            row_start = row * (len(positions) // num_rows)
            for i in range(row_start, row_start + (len(positions) // num_rows)):
                if i % (len(positions) // num_rows) == 0:
                    if row % 2 == 1:
                        delays[i] = delays[i + (len(positions) // num_rows) + 1] + (diagonal_delay if diagonal_delay is not None else 0)
                    else:
                        delays[i] = delays[i + (len(positions) // num_rows)] + (diagonal_delay if diagonal_delay is not None else 0)

                else:
                    delays[i] = delays[i-1] + row_delay

    #if connection_type != 'none' and pattern_type !='square':
        #for i, (x_pos, y_pos) in enumerate(positions):
            #ax.text(x_pos, y_pos, f'{delays[i]} ms' if delays[i] is not None else '',fontsize = 8, ha = 'right')

    ax.grid(False)
    ax.set_xlim(-spacing, max(x) + spacing)
    ax.set_ylim(-spacing, max(y) + burden + 10)
    ax.set_aspect('equal', adjustable = 'box')

    def draw_parallelogram(ax, x_start, y_start, length, height, color='grey', text=None, rotation=0):
        parallelogram = plt.Polygon(
            [(x_start, y_start),
             (x_start + length, y_start),
             (x_start + length, y_start + height),
             (x_start, y_start + height)],
            closed=True, color=color, alpha=0.5)
        ax.add_patch(parallelogram)

        # Add rotated text inside the parallelogram if provided
        if text:
            ax.text(
                x_start + length / 2,
                y_start + height / 2,
                text,
                ha='center', va='center',  # Center alignment remains for rotation
                rotation=rotation,  # Rotate text
                fontsize=10, color='black', fontweight='bold'
            )

    row_length = (len(positions) // num_rows) * spacing

    if 'top' in free_faces:
        # Text for top is not rotated
        draw_parallelogram(ax, 0, max(y) + burden / 2, row_length, burden / 2, text="Front Free Face", rotation=0)

    if 'bottom' in free_faces:
        # Text for bottom is not rotated
        draw_parallelogram(ax, 0, min(y) - burden, row_length, burden / 2, text="Back Free Face", rotation=0)

    if 'left' in free_faces:
        # Text for left is rotated 90 degrees clockwise
        draw_parallelogram(ax, min(x) - burden, min(y), burden / 2, max(y) - min(y), text="Left Free Face", rotation=-90)

    if 'right' in free_faces:
        # Text for right is rotated 90 degrees counterclockwise
        draw_parallelogram(ax, max(x) + burden / 2, min(y), burden / 2, max(y) - min(y), text="Right Free Face", rotation=90)

    arrows = []
    def add_arrow(start_x,start_y,end_x,end_y,color):
        arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,head_width = 0.1, head_length=0.1, fc=color,ec=color)
        arrows.append(arrow)



    if connection_type == 'Diagonal':
        for i in range(len(positions) - 1):
            if y[i] == y[i + 1] and y[i] == (num_rows - 1) * burden:
                add_arrow(x[i], y[i], x[i+1], y[i],'black')

        for row in range(num_rows - 1, 0, -1):
            for i in range(1, len(positions)):
                if y[i] == row * burden:
                    current_x = x[i]
                    current_y = y[i]
                    while True:
                        next_x = current_x - spacing / 2
                        next_y = current_y - burden
                        if (next_x, next_y) in positions:
                            add_arrow(current_x, current_y, next_x, next_y,'red')
                            current_x = next_x
                            current_y = next_y
                        else:
                            break

        for i in range(len(positions) - 1):
            if y[i] == (num_rows - 1) * burden:
                for j in range(len(positions)):
                    if x[j] == x[i] + spacing and y[j] == y[i]:
                        add_arrow(x[i], y[i], x[j],y[j], 'black')

        for i in range(len(positions) - 1):
            if y[i] % (2 * burden) != 0:
                if x[i] == max(x) - spacing:
                    add_arrow(x[i], y[i], x[i] + spacing, y[i], 'black')

        for row in range(num_rows - 2, 0, -1):
            for i in range(len(positions) - 1):
                if y[i] == row * burden and x[i] == max(x):
                    current_x = x[i]
                    current_y = y[i]
                    while True:
                        next_x = current_x - spacing / 2
                        next_y = current_y - burden
                        if (next_x, next_y) in positions:
                            add_arrow(current_x, current_y, next_x, next_y,'red')
                            current_x = next_x
                            current_y = next_y
                        else:
                            break

        for i in range(len(positions) - 1):
            if y[i] == (num_rows - 2) * burden and x[i] == max(x):
                for j in range(len(positions)):
                    if y[j] == (num_rows - 2) * burden and x[j] == x[i] - spacing:
                        add_arrow(x[j], y[j], x[i], y[j], 'black')

        for i in range(len(positions)):
            if y[i] == min(y):
                for j in range(len(positions)):
                    if y[j] == min(y) and x[j] == max(x):
                        add_arrow(x[j - 1], y[j - 1], x[j] , y[j], 'black')

    elif connection_type == 'Line':
        for row in range(num_rows):
            row_positions = [pos for pos in positions if pos[1] == row * burden]
            for i in range(len(row_positions) - 1):
                add_arrow(row_positions[i][0], row_positions[i][1], row_positions[i + 1][0], row_positions[i+1][1],'black')

        for row in range(num_rows - 1, 0, -1):
            for i in range(1, len(positions)):
                if y[i] == row * burden:
                    current_x = x[i]
                    current_y = y[i]
                    next_x = current_x - spacing / 2
                    next_y = current_y - burden
                    if (next_x, next_y) in positions:
                        add_arrow(current_x, current_y, next_x, next_y, 'red')
                        break
    if connection_type != 'None' and pattern_type != 'Square':
        black_arrow = plt.Line2D([0], [0], color='black', lw=2)
        red_arrow = plt.Line2D([0], [0], color='red', lw=2)
        ax.legend([black_arrow, red_arrow], [f'Row wise delay:{row_delay} ms', f'Diagonal delay:{diagonal_delay} ms'],
               loc='upper left')
    return fig, ax, scatter, delays

import plotly.graph_objects as go
import numpy as np


def create_animation_plotly(positions, delays):
    # Validate inputs
    if not isinstance(delays, (list, np.ndarray)) or not all(isinstance(delay, (int, float)) for delay in delays):
        raise TypeError("Invalid delays: Expected a list or array of numeric values.")

    if len(positions) != len(delays):
        raise ValueError("Mismatched positions and delays: lengths must match.")

    # Calculate maximum frame for the animation
    max_frame = int(max(delays)) + 10
    frames = []

    # Extract x and y coordinates from positions
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]

    # Generate frames for Plotly animation
    for frame in range(max_frame):
        # Determine colors and sizes based on frame number and delays
        colors = ['red' if frame >= delay else 'blue' for delay in delays]
        sizes = [40 if frame >= delay else 20 for delay in delays]

        # Append the data for this frame
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(
                        color=colors,
                        size=sizes
                    )
                )
            ],
            name=str(frame)
        ))

    # Create the base figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    color=['blue'] * len(delays),  # Initial colors
                    size=[20] * len(delays)  # Initial sizes
                )
            )
        ],
        layout=go.Layout(

            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        ),
        frames=frames
    )

    # Return the Plotly figure
    return fig


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    mine_name = request.form['mine_name']
    location = request.form['location']
    date_str = request.form['date']
    free_faces = request.form.getlist('free_face[]')
    time_str = request.form['time']
    Latitude = float(request.form['Latitude'])
    Longitude = float(request.form['Longitude'])
    pattern_type= request.form['pattern_type']
    deck_charging = request.form.get('deck_charging', 'none')
    num_decked_holes = int(request.form['num_decked_holes']) if request.form.get('deck_charging'
                                                                                 '').lower() == "yes" else 0
    connection_type= request.form.get('connection_type', 'none')
    num_holes = int(request.form['num_holes'])
    burden = float(request.form['burden'])
    spacing = float(request.form['spacing'])
    num_rows = int(request.form['num_rows'])
    diameter_mm = float(request.form['diameter_mm'])
    depth_m = float(request.form['depth_m'])
    explosive_type = request.form['explosive_type']
    explosive_density_g_cm3 = float(request.form['explosive_density_g_cm3'])
    total_explosive_quantity_kg = float(request.form['total_explosive_quantity_kg'])
    nonel_length_m = float(request.form['nonel_length_m'])
    booster_quantity_g = float(request.form['booster_quantity_g'])
    rock_density = float(request.form['rock_density'])
    explosive_quantity_top_kg = float(request.form['explosive_quantity_top_kg']) if request.form.get('deck_charging',
                                                                                                     '').lower() == "yes" else 0.0
    explosive_quantity_bottom_kg = float(request.form['explosive_quantity_bottom_kg']) if request.form.get(
        'deck_charging',
        '').lower() == "yes" else 0.0
    nonel_length_top_m = float(request.form['nonel_length_top_m']) if request.form.get('deck_charging',
                                                                                       '').lower() == "yes" else 0.0
    nonel_length_bottom_m = float(request.form['nonel_length_bottom_m']) if request.form.get('deck_charging',
                                                                                             '').lower() == "yes" else 0.0

    distance = float(request.form['distance'])
    k_constant = float(request.form['k_constant'])
    e_constant = float(request.form['e_constant'])
    row_delay = float(request.form.get('row_delay',0) or 0)
    diagonal_delay = float(request.form.get('diagonal_delay',0) or 0)
    user_input =request.form['user_input']
    explosive_cost_kg = float(request.form['explosive_cost_kg'])
    booster_cost_kg = float(request.form['booster_cost_kg'])
    nonel_cost_m = float(request.form['nonel_cost_m'])
    electronic_detonators = int(request.form['electronic_detonators'])
    electrical_detonators = int(request.form['electrical_detonators'])
    explosive_quantity_kg = total_explosive_quantity_kg / num_holes
    #total_booster_quantity_g = booster_quantity_g *num_holes
    if deck_charging == 'Yes':
        total_booster_quantity_g = booster_quantity_g * (num_holes + num_decked_holes )
    else:
        total_booster_quantity_g = booster_quantity_g * num_holes

    volume_of_patch_m3 = depth_m*spacing*burden*num_holes
    powder_factor = volume_of_patch_m3/(total_explosive_quantity_kg + total_booster_quantity_g/1000)
    charge_per_hole = explosive_quantity_kg + booster_quantity_g/1000
    ppv = calculate_ppv(distance,charge_per_hole,k_constant,e_constant)
    mean_fragmentation_size = 12.4 * (burden*spacing*depth_m/charge_per_hole )**0.8* charge_per_hole **0.167
    total_explosive_cost= total_explosive_quantity_kg*explosive_cost_kg
    total_booster_cost = (total_booster_quantity_g / 1000) *booster_cost_kg
    total_nonel_length = 0
    if pattern_type == 'Staggered' and connection_type != 'None':
        total_nonel_length = (num_holes * spacing) + (num_holes * nonel_length_m)

    total_blasting_cost = total_explosive_cost + total_booster_cost + total_nonel_length * nonel_cost_m
    explosive_density_kg_m3 = explosive_density_g_cm3 * 1000
    charge_height = explosive_quantity_kg / (explosive_density_kg_m3 * (diameter_mm / 1000) ** 2 * 3.141592653589793 / 4)
    stemming_distance_m = depth_m - charge_height
    post_blast_image = request.files.get('post_blast_image')
    post_blast_image_base64 = None

    if post_blast_image and post_blast_image.filename != '':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], post_blast_image.filename)
        post_blast_image.save(image_path)  # Save image to the upload folder

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Define the text to be added
        text = (f"Mine Name: {mine_name}\n"
                f"Date : {date_str}\n"
                f"Time : {time_str}\n"
                f"Location: {location}\n"
                f"Latitude : {Latitude}\n"
                f"Longitude : {Longitude}")

        # Calculate the text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Image dimensions
        img_width, img_height = img.size

        # Define dynamic padding (for proportional scaling)
        padding = max(img_width, img_height) // 30  # Adjust 1/50th of the largest dimension

        # Define the text position dynamically (e.g., 5% inset from top-left)
        x_offset = img_width // 20  # 5% inset from left
        y_offset = img_height // 20  # 5% inset from top
        text_position = (x_offset, y_offset)

        # Define the box coordinates dynamically
        box_coords = [
            (text_position[0] - padding, text_position[1] - padding),  # Top-left of the box
            (text_position[0] + text_width + padding, text_position[1] + text_height + padding)
            # Bottom-right of the box
        ]

        # Draw the background rectangle with padding (semi-transparent if needed)
        draw.rectangle(box_coords, fill="white", outline="black")

        # Draw the text on top of the rectangle
        draw.multiline_text(text_position, text, fill="black", font=font)

        # Convert the image to Base64
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        post_blast_image_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

    #Generate summary table
    data_summary = {
        'SPECIFICATIONS': [
            'Mine Name',
            'Date',
            'Time',
            'Location',
            'Latitude',
            'Longitude',
            'Number of Holes',
            'Number of Decked Holes',
            'Average Spacing (m)',
            'Average Burden (m)',
            'Hole Diameter (mm)',
            'Average Hole Depth (m)',
            'Explosive Type',
            'Explosive Density (g/cm3)',
            'Pattern Type',
            'Connection Type',
            'Row Delay (ms)',
            'Diagonal Delay (ms)',
            'Average Explosive Quantity per Hole(Kg)',
            'Total Explosive Quantity (Kg)',
            'Total Booster Quantity (Kg)',
            'Electronic Detonators Used',
            'Electrical Detonators Used',
            'Volume of Patch (m3)',
            'Powder Factor (PF)',
            'Average Stemming Distance (m)',
            'Average Charge Height (m)',
            'PPV (Peak Particle Velocity) (mm/s)',
            'Mean Fragmentation Size (cm)',

        ],
        'DESCRIPTION':[
            mine_name,
            date_str,
            time_str,
            location,
            Latitude,
            Longitude,
            num_holes,
            num_decked_holes,
            spacing,
            burden,
            diameter_mm,
            depth_m,
            explosive_type,
            explosive_density_g_cm3,
            pattern_type,
            connection_type,
            row_delay,
            diagonal_delay,
            round(explosive_quantity_kg,0),
            round(total_explosive_quantity_kg,3),
            round(total_booster_quantity_g/1000,3),
            electronic_detonators,
            electrical_detonators,
            round(volume_of_patch_m3,3),
            round(powder_factor,3),
            round(stemming_distance_m,2),
            round(charge_height,2),
            round(ppv,3),
            round(mean_fragmentation_size, 3)


         ]
    }
    df_summary = pd.DataFrame(data_summary)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns, cellLoc='center', loc='center',
                     colColours=['#4CAF50', '#FF9800'], cellColours=[['#E8F5E9', '#FFF3E0']] * len(df_summary))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_fontsize(10)
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
    
    plt.title('Blasting Summary',fontsize=16,fontweight='bold',color='#4CAF50')
    summary_table_img = BytesIO()
    plt.savefig(summary_table_img, format='png')
    summary_table_img.seek(0)
    summary_table_img_base64 = base64.b64encode(summary_table_img.read()).decode('utf-8')

    positions = generate_blasting_pattern(pattern_type, num_holes, burden, spacing, num_rows)
    fig,ax,scatter,delays = plot_blasting_pattern(positions,burden,spacing,num_rows,connection_type,row_delay=row_delay,diagonal_delay=diagonal_delay,free_faces= free_faces)
    animation_html = None
    blasting_pattern_base64 = None
    if user_input == 'yes':
        anim_fig = create_animation_plotly(positions, delays)
        animation_html = anim_fig.to_html(full_html=False)
    else:
        blasting_pattern_img = BytesIO()
        plt.savefig(blasting_pattern_img, format='png')
        blasting_pattern_img.seek(0)
        #blasting_pattern_base64 = base64.b64encode(blasting_pattern_img.read()).decode('utf-8')
        # Use PIL for image processing
        img = Image.open(blasting_pattern_img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        ## Define the text to be added
        text = (f"Mine Name: {mine_name}\n"
                f"Date : {date_str}\n"
                f"Time : {time_str}\n"
                f"Location: {location}\n"
                f"Latitude : {Latitude}\n"
                f"Longitude : {Longitude}")

        # Calculate the text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Image dimensions
        img_width, img_height = img.size

        # Define dynamic padding (for proportional scaling)
        padding = max(img_width, img_height) // 20  # Adjust 1/50th of the largest dimension

        # Define the text position dynamically (e.g., 5% inset from top-left)
        x_offset = img_width // 20  # 5% inset from left
        y_offset = img_height // 20  # 5% inset from top
        text_position = (x_offset, y_offset)

        # Define the box coordinates dynamically
        box_coords = [
            (text_position[0] - padding, text_position[1] - padding),  # Top-left of the box
            (text_position[0] + text_width + padding, text_position[1] + text_height + padding)
            # Bottom-right of the box
        ]

        # Draw the background rectangle with padding (semi-transparent if needed)
        draw.rectangle(box_coords, fill="white", outline="black")

        # Draw the text on top of the rectangle
        draw.multiline_text(text_position, text, fill="black", font=font)

        # Convert the annotated image to Base64
        annotated_img_byte_arr = BytesIO()
        img.save(annotated_img_byte_arr, format='PNG')
        annotated_img_byte_arr.seek(0)
        blasting_pattern_base64 = base64.b64encode(annotated_img_byte_arr.read()).decode('utf-8')

        # Clean up Matplotlib to avoid memory issues
        plt.close(fig)

    #blasting_pattern_img = BytesIO()
    #plt.savefig(blasting_pattern_img, format='png')
    #blasting_pattern_img.seek(0)
    #blasting_pattern_base64 = base64.b64encode(blasting_pattern_img.read()).decode('utf-8')

    explosive_density_kg_m3 = explosive_density_g_cm3 * 1000
    if deck_charging == 'Yes' :
        if num_holes == num_decked_holes:
            direct_charge_explosive_quantity = 0
        else:
            direct_charge_explosive_quantity = (total_explosive_quantity_kg - (num_decked_holes * (explosive_quantity_top_kg + explosive_quantity_bottom_kg))) / (num_holes - num_decked_holes)
    else:
       direct_charge_explosive_quantity = explosive_quantity_kg
   
    charge_height = direct_charge_explosive_quantity / (explosive_density_kg_m3 * (diameter_mm / 1000) ** 2 * 3.141592653589793 / 4)
    stemming_distance_m = depth_m - charge_height
    
    fig, ax = plt.subplots()
    charge = plt.Rectangle((0.5-diameter_mm/2000, depth_m- charge_height), diameter_mm/1000,charge_height, edgecolor='black',facecolor='black', label='Explosive Charge')
    ax.add_patch(charge)
    stemming = plt.Rectangle((0.5-diameter_mm/2000,0), diameter_mm /1000, stemming_distance_m, edgecolor='black',facecolor='grey', label='Stemming Distance')
    ax.add_patch(stemming)
    void_space_height = depth_m- charge_height - stemming_distance_m
    void_space = plt.Rectangle((0.5 - diameter_mm/ 2000, stemming_distance_m), diameter_mm/ 1000, void_space_height, edgecolor='black',facecolor='none', label='Void Space')
    ax.add_patch(void_space)
    nonel_line_length =nonel_length_m
    nonel_line = plt.Line2D([0.5] * 2, [depth_m- nonel_line_length, depth_m - 0.2], color='red', linewidth = 2, label='Nonel Line')
    ax.add_line(nonel_line)
    booster_square = plt.Rectangle((0.5 - diameter_mm/2000 /2, depth_m-0.7), 0.5*diameter_mm/1000, 0.5, edgecolor = 'black' , facecolor ='yellow', label = 'Booster')
    ax.add_patch(booster_square)
    arrowprops = dict(facecolor='black', shrink=0.05, width = 1)
    ax.annotate(f'Depth: {depth_m} m', xy=(0.5+ diameter_mm / 2000 / 2, depth_m),xytext=(1.5, depth_m),arrowprops=arrowprops, ha='center')
    
    ax.annotate(f'Charge Height:{charge_height:.2f} m', xy=(0.5 + diameter_mm/ 2000 /2, depth_m - charge_height / 2), xytext = (1.5, depth_m - charge_height / 2), arrowprops=dict(facecolor='black',shrink = 0.05, width = 1), ha='center', color = 'black')
    
    ax.annotate(f'Stemming Distance:{stemming_distance_m:.2f} m', xy=(0.5 + diameter_mm / 2000 /2, stemming_distance_m/ 2), xytext = (1.5, stemming_distance_m/ 2), arrowprops=dict(facecolor='grey',shrink = 0.05, width = 1), ha='center', color = 'black')
    
    ax.set_ylim(depth_m+ 1, -1)
    ax.set_xlim(0, 3)
    #plt.title('Single Hole Diagram')
    plt.legend(loc='upper right')
    single_hole_diagram_img= BytesIO()
    plt.savefig(single_hole_diagram_img,format='png')
    single_hole_diagram_img.seek(0)
    single_hole_diagram_base64 = base64.b64encode(single_hole_diagram_img.getvalue()).decode('utf-8')

    # Calculate charge height for each explosive
    charge_height_top = explosive_quantity_top_kg / (
                explosive_density_kg_m3 * (diameter_mm / 1000) ** 2 * 3.141592653589793 / 4)
    charge_height_bottom = explosive_quantity_bottom_kg / (
                explosive_density_kg_m3 * (diameter_mm / 1000) ** 2 * 3.141592653589793 / 4)

    # Calculate stemming distances
    total_charge_height = charge_height_top + charge_height_bottom
    intermediate_stemming_m = (depth_m - total_charge_height) / 2
    top_stemming_m = depth_m - (charge_height_top + charge_height_bottom + intermediate_stemming_m)

    # Create combined plot
    fig, ax = plt.subplots()

    # Plot bottom explosive charge
    charge_bottom = plt.Rectangle((0.5 - diameter_mm / 2000, 0), diameter_mm / 1000, charge_height_bottom,
                                  edgecolor='black', facecolor='black')
    ax.add_patch(charge_bottom)

    # Plot intermediate stemming
    stemming_intermediate = plt.Rectangle((0.5 - diameter_mm / 2000, charge_height_bottom), diameter_mm / 1000,
                                          intermediate_stemming_m, edgecolor='black', facecolor='grey')
    ax.add_patch(stemming_intermediate)

    # Plot top explosive charge
    charge_top = plt.Rectangle((0.5 - diameter_mm / 2000, charge_height_bottom + intermediate_stemming_m),
                               diameter_mm / 1000, charge_height_top, edgecolor='black', facecolor='black',
                               label='Explosive Charge')
    ax.add_patch(charge_top)

    # Plot top stemming
    stemming_top = plt.Rectangle(
        (0.5 - diameter_mm / 2000, charge_height_bottom + intermediate_stemming_m + charge_height_top),
        diameter_mm / 1000, top_stemming_m, edgecolor='black', facecolor='grey', label='Stemming')
    ax.add_patch(stemming_top)

    # Plot Nonel lines (reversed)
    nonel_line_bottom = plt.Line2D([0.5] * 2,
                                   [charge_height_bottom / 2, charge_height_bottom / 2 + nonel_length_bottom_m],
                                   color='red', linewidth=2)
    ax.add_line(nonel_line_bottom)
    nonel_line_top = plt.Line2D([0.47] * 2, [charge_height_bottom + intermediate_stemming_m + charge_height_top / 2,
                                             charge_height_bottom + intermediate_stemming_m + charge_height_top / 2 + nonel_length_top_m],
                                color='red', linewidth=2, label='Nonel Line')
    ax.add_line(nonel_line_top)

    # Plot boosters
    booster_bottom = plt.Rectangle((0.5 - diameter_mm / 2000 / 2, charge_height_bottom / 7), 0.5 * diameter_mm / 1000,
                                   0.5, edgecolor='black', facecolor='yellow')
    ax.add_patch(booster_bottom)
    booster_top = plt.Rectangle(
        (0.47 - diameter_mm / 2000 / 2, charge_height_bottom + intermediate_stemming_m + charge_height_top / 7),
        0.5 * diameter_mm / 1000, 0.5, edgecolor='black', facecolor='yellow', label='Booster')
    ax.add_patch(booster_top)

    # Add annotations
    arrowprops = dict(facecolor='black', shrink=0.05, width=1)
    ax.annotate(f'Bottom Charge Height: {charge_height_bottom:.2f} m',
                xy=(0.5 + diameter_mm / 2000 / 2, charge_height_bottom / 2), xytext=(1.5, charge_height_bottom / 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1), ha='center', color='black')
    ax.annotate(f'Intermediate Stemming: {intermediate_stemming_m:.2f} m',
                xy=(0.5 + diameter_mm / 2000 / 2, charge_height_bottom + intermediate_stemming_m / 2),
                xytext=(1.5, charge_height_bottom + intermediate_stemming_m / 2),
                arrowprops=dict(facecolor='grey', shrink=0.05, width=1), ha='center', color='black')
    ax.annotate(f'Top Charge Height: {charge_height_top:.2f} m', xy=(
    0.5 + diameter_mm / 2000 / 2, charge_height_bottom + intermediate_stemming_m + charge_height_top / 2),
                xytext=(1.5, charge_height_bottom + intermediate_stemming_m + charge_height_top / 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1), ha='center', color='black')
    ax.annotate(f'Top Stemming: {top_stemming_m:.2f} m', xy=(0.5 + diameter_mm / 2000 / 2,
                                                              charge_height_bottom + intermediate_stemming_m + charge_height_top + top_stemming_m / 2),
                xytext=(1.5, charge_height_bottom + intermediate_stemming_m + charge_height_top + top_stemming_m / 2),
                arrowprops=dict(facecolor='grey', shrink=0.05, width=1), ha='center', color='black')

    # Set plot limits and legend
    ax.set_ylim(-1, depth_m + 1)
    ax.set_xlim(0, 3)
    plt.legend(loc='upper right', fontsize='small')

    # Save plot as image
    combined_hole_diagram_img = BytesIO()
    plt.savefig(combined_hole_diagram_img, format='png')
    combined_hole_diagram_img.seek(0)
    combined_hole_diagram_base64 = base64.b64encode(combined_hole_diagram_img.getvalue()).decode('utf-8')

    #animation_html = None
    #blasting_pattern_base64 = None  # Ensure it's initialized

    #if user_input == 'yes':
        #anim = create_animation(fig, ax, scatter, delays)
        #animation_html = anim.to_jshtml()
    #else:
        #blasting_pattern_img = BytesIO()
        #plt.savefig(blasting_pattern_img, format='png')
        #blasting_pattern_img.seek(0)
        #blasting_pattern_base64 = base64.b64encode(blasting_pattern_img.read()).decode('utf-8')



    #animation_html = None
    #if user_input == 'yes':
        #anim = create_animation(fig, ax, scatter, delays)
        #animation_html = anim.to_jshtml()

    if deck_charging.lower() == "yes":
        if num_holes == num_decked_holes:
            # Generate the combined hole diagram only
            combined_hole_diagram_img = BytesIO()
            plt.savefig(combined_hole_diagram_img, format='png')
            combined_hole_diagram_img.seek(0)
            combined_hole_diagram_base64 = base64.b64encode(combined_hole_diagram_img.getvalue()).decode('utf-8')
            single_hole_diagram_base64 = None  # No single hole diagram in this case
        elif (num_holes > num_decked_holes) and (num_holes < num_decked_holes):
            # Generate both single and combined diagrams
            # Save Single Hole Diagram
            single_hole_diagram_img = BytesIO()
            plt.savefig(single_hole_diagram_img, format='png')
            single_hole_diagram_img.seek(0)
            single_hole_diagram_base64 = base64.b64encode(single_hole_diagram_img.getvalue()).decode('utf-8')

            # Save Combined Hole Diagram
            combined_hole_diagram_img = BytesIO()
            plt.savefig(combined_hole_diagram_img, format='png')
            combined_hole_diagram_img.seek(0)
            combined_hole_diagram_base64 = base64.b64encode(combined_hole_diagram_img.getvalue()).decode('utf-8')
    elif deck_charging.lower() == "no":
        combined_hole_diagram_base64 = None





    return render_template('plot.html',summary_table= df_summary.values,blasting_pattern=blasting_pattern_base64,single_hole_diagram=single_hole_diagram_base64,combined_hole_diagram=combined_hole_diagram_base64,animation_html=animation_html,post_blast_image=post_blast_image_base64)

if __name__ == '__main__':
    db.create_all()
    app.run(debug= True )
