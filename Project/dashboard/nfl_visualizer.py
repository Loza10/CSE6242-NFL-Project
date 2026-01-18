##AMIR'S CODE/SCRIPT
"""
NFL Play Visualizer - Complete standalone file
Shows player movements during pass plays with ball tracking
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


def create_play_animation(play_df, play_id, show_ball=True):
    """
    Create animated visualization of an NFL play.
    
    Args:
        play_df: DataFrame with tracking data for one play
        play_id: Play ID
        show_ball: If True, show ball position
        
    Returns:
        Plotly Figure
    """
    # Get metadata
    num_frames_output = play_df['num_frames_output'].iloc[0]
    ball_land_x = play_df['ball_land_x'].iloc[0]
    ball_land_y = play_df['ball_land_y'].iloc[0]
    
    # Get frames
    frames_list = sorted(play_df['frame_id'].unique())
    total_frames = len(frames_list)
    throw_frame = total_frames - num_frames_output + 1
    
    # Get QB position for ball start
    qb_data = play_df[play_df['player_position'].str.contains('QB', na=False)]
    if not qb_data.empty:
        qb_throw = qb_data[qb_data['frame_id'] == throw_frame]
        if not qb_throw.empty:
            ball_start_x = qb_throw.iloc[0]['x']
            ball_start_y = qb_throw.iloc[0]['y']
        else:
            ball_start_x = qb_data.iloc[0]['x']
            ball_start_y = qb_data.iloc[0]['y']
    else:
        ball_start_x = 60
        ball_start_y = 26.67
    
    def get_ball_position(frame_id):
        """Calculate ball position for a frame"""
        if frame_id < throw_frame:
            qb_frame = qb_data[qb_data['frame_id'] == frame_id]
            if not qb_frame.empty:
                return qb_frame.iloc[0]['x'], qb_frame.iloc[0]['y']
            return ball_start_x, ball_start_y
        else:
            progress = (frame_id - throw_frame) / (total_frames - throw_frame)
            x = ball_start_x + progress * (ball_land_x - ball_start_x)
            y = ball_start_y + progress * (ball_land_y - ball_start_y)
            return x, y
    
    # Separate teams
    offense = play_df[play_df['player_side'] == 'Offense']
    defense = play_df[play_df['player_side'] == 'Defense']
    
    # Create field shapes
    field_shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=53.33,
             line=dict(color='white', width=3), fillcolor='#2d5016', layer='below'),
        dict(type="line", x0=10, y0=0, x1=10, y1=53.33,
             line=dict(color='white', width=4), layer='below'),
        dict(type="line", x0=110, y0=0, x1=110, y1=53.33,
             line=dict(color='white', width=4), layer='below'),
    ]
    
    for yard in range(20, 110, 10):
        field_shapes.append(
            dict(type="line", x0=yard, y0=0, x1=yard, y1=53.33,
                 line=dict(color='white', width=2, dash='dot'), layer='below')
        )
    
    # Create figure
    fig = go.Figure()
    
    # Initial frame
    initial_frame = frames_list[0]
    off_init = offense[offense['frame_id'] == initial_frame]
    def_init = defense[defense['frame_id'] == initial_frame]
    ball_init_x, ball_init_y = get_ball_position(initial_frame)
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=off_init['x'], y=off_init['y'],
        mode='markers+text', name='Offense',
        marker=dict(size=12, color='#0173b2', symbol='circle', line=dict(width=2, color='white')),
        text=off_init['player_position'],
        textposition='top center', textfont=dict(size=8, color='white'),
        hovertext=off_init['player_name'], hoverinfo='text'
    ))
    
    fig.add_trace(go.Scatter(
        x=def_init['x'], y=def_init['y'],
        mode='markers+text', name='Defense',
        marker=dict(size=12, color='#de8f05', symbol='square', line=dict(width=2, color='white')),
        text=def_init['player_position'],
        textposition='top center', textfont=dict(size=8, color='white'),
        hovertext=def_init['player_name'], hoverinfo='text'
    ))
    
    if show_ball:
        fig.add_trace(go.Scatter(
            x=[ball_init_x], y=[ball_init_y],
            mode='markers+text', name='🏈 Ball',
            marker=dict(size=14, color='#8B4513', symbol='diamond', line=dict(width=3, color='white')),
            text=['🏈'], textfont=dict(size=16), textposition='middle center',
            hovertext='Football', hoverinfo='text'
        ))
    
    fig.add_trace(go.Scatter(
        x=[ball_land_x], y=[ball_land_y],
        mode='markers', name='🎯 Target',
        marker=dict(size=15, color='red', symbol='x', line=dict(width=3)),
        hovertext='Ball lands here', hoverinfo='text'
    ))
    
    # Create animation frames
    animation_frames = []
    for frame_id in frames_list:
        off_frame = offense[offense['frame_id'] == frame_id]
        def_frame = defense[defense['frame_id'] == frame_id]
        ball_x, ball_y = get_ball_position(frame_id)
        is_airborne = frame_id >= throw_frame
        status = "🏈 BALL IN AIR" if is_airborne else "Pre-throw"
        
        frame_data = [
            go.Scatter(
                x=off_frame['x'], y=off_frame['y'], mode='markers+text',
                marker=dict(size=12, color='#0173b2', symbol='circle', line=dict(width=2, color='white')),
                text=off_frame['player_position'], textposition='top center',
                textfont=dict(size=8, color='white'),
                hovertext=off_frame['player_name'], hoverinfo='text'
            ),
            go.Scatter(
                x=def_frame['x'], y=def_frame['y'], mode='markers+text',
                marker=dict(size=12, color='#de8f05', symbol='square', line=dict(width=2, color='white')),
                text=def_frame['player_position'], textposition='top center',
                textfont=dict(size=8, color='white'),
                hovertext=def_frame['player_name'], hoverinfo='text'
            ),
        ]
        
        if show_ball:
            frame_data.append(go.Scatter(
                x=[ball_x], y=[ball_y], mode='markers+text',
                marker=dict(size=14, color='#8B4513', symbol='diamond', line=dict(width=3, color='white')),
                text=['🏈'], textfont=dict(size=16), textposition='middle center',
                hovertext='Football', hoverinfo='text'
            ))
        
        frame_data.append(go.Scatter(
            x=[ball_land_x], y=[ball_land_y], mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=3)),
            hovertext='Ball lands here', hoverinfo='text'
        ))
        
        animation_frames.append(go.Frame(
            data=frame_data, name=str(frame_id),
            layout=go.Layout(title_text=f"Play {play_id} - Frame {frame_id}/{total_frames} - {status}")
        ))
    
    fig.frames = animation_frames
    
    # Layout
    fig.update_layout(
        title=f"Play {play_id} - Frame {initial_frame}/{total_frames}",
        plot_bgcolor='#2d5016', height=600, width=1200,
        shapes=field_shapes,
        xaxis=dict(range=[0, 120], showgrid=False, title="Field Position (yards)", zeroline=False),
        yaxis=dict(range=[0, 53.33], showgrid=False, title="Width (yards)",
                  scaleanchor="x", scaleratio=1, zeroline=False),
        updatemenus=[{
            'type': 'buttons', 'showactive': False,
            'buttons': [
                {'label': '▶ Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True, 'mode': 'immediate'}]},
                {'label': '⏸ Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate'}]}
            ], 'x': 0.1, 'y': 1.15
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate'}],
                 'label': f"{frame_id} {'🏈' if frame_id >= throw_frame else ''}",
                 'method': 'animate'}
                for frame_id, f in zip(frames_list, animation_frames)
            ],
            'x': 0.1, 'len': 0.85, 'y': 0
        }],
        showlegend=True, hovermode='closest'
    )
    
    return fig


if __name__ == "__main__":
    # Load data
    print("🏈 Loading NFL data...")
    df = pd.read_csv('data/input_2023_w01.csv')
    
    # Pick a play
    play_id = 302
    play_df = df[df['play_id'] == play_id]
    
    print(f"📊 Creating animation for Play {play_id}...")
    print(f"   Total frames: {play_df['frame_id'].max()}")
    print(f"   Ball airborne: {play_df['num_frames_output'].iloc[0]} frames")
    
    # Create animation
    fig = create_play_animation(play_df, play_id)
    
    # Save
    Path('outputs').mkdir(exist_ok=True)
    output_file = 'outputs/play_animation.html'
    fig.write_html(output_file)
    print(f"✅ Saved to: {output_file}")
    
    # Show
    fig.show()
    print("\n👆 Use slider to scrub frames, or press Play!")