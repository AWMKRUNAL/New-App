import base64
import os
from io import BytesIO
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
from flask import Flask, render_template
from flask import request
from werkzeug.security import generate_password_hash, check_password_hash
from matplotlib.animation import FuncAnimation
import mysql.connector

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
            if pattern_type == 'square':
                positions.append((j * spacing, i * spacing))
            elif pattern_type == 'staggered':
                x_offset = j * spacing + (spacing / 2 if i % 2 == 1 else 0)
                positions.append((x_offset, i * burden))
            else:
                raise ValueError("Unsupported Pattern type. Use 'square' or 'staggered'.")
    return positions


def plot_blasting_pattern(positions, burden, spacing, num_rows, connection_type, row_delay=None, diagonal_delay=None,
                          pattern_type=None):
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


    arrows = []
    def add_arrow(start_x,start_y,end_x,end_y,color):
        arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,head_width = 0.1, head_length=0.1, fc=color,ec=color)
        arrows.append(arrow)



    if connection_type == 'diagonal':
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

    elif connection_type == 'line':
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
    if connection_type != 'none' and pattern_type != 'square':
        black_arrow = plt.Line2D([0], [0], color='black', lw=2)
        red_arrow = plt.Line2D([0], [0], color='red', lw=2)
        ax.legend([black_arrow, red_arrow], [f'Row wise delay:{row_delay} ms', f'Diagonal delay:{diagonal_delay} ms'],
               loc='upper left')
    return fig, ax, scatter, delays

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
    pattern_type= request.form['pattern_type']
    connection_type= request.form.get('connection_type', 'none')
    num_holes = int(request.form['num_holes'])
    burden = float(request.form['burden'])
    spacing = float(request.form['spacing'])
    num_rows = int(request.form['num_rows'])
    diameter_mm = float(request.form['diameter_mm'])
    depth_m = float(request.form['depth_m'])
    explosive_type = request.form['explosive_type']
    explosive_density_g_cm3 = float(request.form['explosive_density_g_cm3'])
    explosive_quantity_kg = float(request.form['explosive_quantity_kg'])
    nonel_length_m = float(request.form['nonel_length_m'])
    booster_quantity_g = float(request.form['booster_quantity_g'])
    rock_density = float(request.form['rock_density'])
    distance = float(request.form['distance'])
    k_constant = float(request.form['k_constant'])
    e_constant = float(request.form['e_constant'])
    row_delay = float(request.form.get('row_delay',0) or 0)
    diagonal_delay = float(request.form.get('diagonal_delay',0) or 0)
    user_input =request.form['user_input']
    explosive_cost_kg = float(request.form['explosive_cost_kg'])
    booster_cost_kg = float(request.form['booster_cost_kg'])
    nonel_cost_m = float(request.form['nonel_cost_m'])
    total_explosive_quantity_kg = explosive_quantity_kg*num_holes
    total_booster_quantity_g = booster_quantity_g*num_holes
    volume_of_patch_m3 = depth_m*spacing*burden*num_holes
    powder_factor = volume_of_patch_m3/(total_explosive_quantity_kg + total_booster_quantity_g/1000)
    charge_per_hole = explosive_quantity_kg + booster_quantity_g/1000
    ppv = calculate_ppv(distance,charge_per_hole,k_constant,e_constant)
    mean_fragmentation_size = 8 * (burden*spacing*depth_m/charge_per_hole )**0.8* charge_per_hole **0.167
    total_explosive_cost= total_explosive_quantity_kg*explosive_cost_kg
    total_booster_cost = (total_booster_quantity_g / 1000) *booster_cost_kg
    total_nonel_length = 0
    if pattern_type == 'staggered' and connection_type != 'none':
        total_nonel_length = (num_holes * spacing) + (num_holes * nonel_length_m)

    total_blasting_cost = total_explosive_cost + total_booster_cost + total_nonel_length * nonel_cost_m

    post_blast_image = request.files.get('post_blast_image')
    post_blast_image_base64 = None

    if post_blast_image and post_blast_image.filename != '':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], post_blast_image.filename)
        post_blast_image.save(image_path)  # Save image to the upload folder

        # Read and encode the image to base64
        with open(image_path, 'rb') as f:
            img_byte_stream = f.read()
            post_blast_image_base64 = base64.b64encode(img_byte_stream).decode('utf-8')

#Generate summary table
    data_summary = {
        'SPECIFICATIONS': [
            'Mine Name',
            'Location',
            'Date',
            'Number of Holes',
            'Spacing (m)',
            'Burden (m)',
            'Hole Diameter (mm)',
            'Hole Depth (m)',
            'Explosive Type',
            'Total Explosive Quantity (Kg)',
            'Total Booster Quantity (g)',
            'Volume of Patch (m3)',
            'Powder Factor (PF)',
            'Ideal Charge per Hole (Kg)',
            'PPV(Peak Particle Velocity) (mm/s)',
            'Mean Fragmentation Size (cm)',
            'Total Blasting Cost (₹)'
        ],
        'DESCRIPTION':[
            mine_name,
            location,
            date_str,
            num_holes,
            spacing,
            burden,
            diameter_mm,
            depth_m,
            explosive_type,
            round(total_explosive_quantity_kg,3),
            round(total_booster_quantity_g,3),
            round(volume_of_patch_m3,3),
            round(powder_factor,3),
            round(charge_per_hole,3),
            round(ppv,3),
            round(mean_fragmentation_size, 3),
            total_blasting_cost


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
    fig,ax,scatter,delays = plot_blasting_pattern(positions,burden,spacing,num_rows,connection_type,row_delay=row_delay,diagonal_delay=diagonal_delay)
    animation_html = None
    blasting_pattern_base64 = None
    if user_input == 'yes':
        anim_fig = create_animation_plotly(positions, delays)
        animation_html = anim_fig.to_html(full_html=False)
    else:
        blasting_pattern_img = BytesIO()
        plt.savefig(blasting_pattern_img, format='png')
        blasting_pattern_img.seek(0)
        blasting_pattern_base64 = base64.b64encode(blasting_pattern_img.read()).decode('utf-8')
    #blasting_pattern_img = BytesIO()
    #plt.savefig(blasting_pattern_img, format='png')
    #blasting_pattern_img.seek(0)
    #blasting_pattern_base64 = base64.b64encode(blasting_pattern_img.read()).decode('utf-8')
    
    explosive_density_kg_m3 = explosive_density_g_cm3 * 1000
    charge_height = explosive_quantity_kg / (explosive_density_kg_m3 * (diameter_mm / 1000) ** 2 * 3.141592653589793 / 4)
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
    nonel_line = plt.Line2D([0.5] * 2, [depth_m- nonel_line_length, depth_m - 0.2], color='orange', linewidth = 2, label='Nonel Line')
    ax.add_line(nonel_line)
    booster_square = plt.Rectangle((0.5 - diameter_mm/2000 /2, depth_m-0.2), diameter_mm/1000, 0.2, edgecolor = 'black' , facecolor ='yellow', label = 'Booster')
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
    return render_template('plot.html',summary_table= df_summary.values,blasting_pattern=blasting_pattern_base64,single_hole_diagram=single_hole_diagram_base64,animation_html=animation_html,post_blast_image=post_blast_image_base64)

if __name__ == '__main__':
    app.run()
