import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


st.title("Player Trajectory Plotting")

df_pre = pd.read_csv("data/input_2023_w01.csv") #Can be swapped with any other week to generate different visuals

df_post = pd.read_csv("data/output_2023_w01.csv") #Can be swapped with any other week to generate different visals

game_id = 2023101900

play_id  = 610

# Dropdowns Lists
unique_games = sorted(df_pre['game_id'].unique())
game_id = st.selectbox("Select Game ID", unique_games)

# Get plays for the selected game
available_plays = sorted(df_pre[df_pre['game_id'] == game_id]['play_id'].unique())
play_id = st.selectbox("Select Play ID", available_plays)

# Westley's Visualization Function, slightly changed
def plot_pre_post(pre_data, post_data, game_id, play_id):
    """Plot pre- and post-throw data (one set, actual or predicted) for a given game_id and play_id"""
    
    pos_map = {'Passer':{'color':'Greys',
                         'marker_pre':'o',
                         'marker_post':'.'
                        },
               'Targeted Receiver':{'color':'Greens',
                                    'marker_pre':'o',
                                    'marker_post':'.'
                                   },
               'Other Route Runner':{'color':'Blues',
                                     'marker_pre':'o',
                                     'marker_post':'.'
                                    },
               'Defensive Coverage':{'color':'Reds',
                                     'marker_pre':'X',
                                     'marker_post':'x'
                                    },
              }
    
    # define custom legend elements
    handles = [Line2D([0], [0], marker='o', linestyle='none', color='black', label='Passer'),
               Line2D([0], [0], marker='o', linestyle='none', color='green', label='Targeted Receiver'),
               Line2D([0], [0], marker='.', linestyle='none', color='green', label='Targeted Receiver (Post-throw)'),
               Line2D([0], [0], marker='o', linestyle='none', color='blue', label='Other Route Runner'),
               Line2D([0], [0], marker='X', linestyle='none', color='red', markersize=7, label='Defensive Coverage'),
               Line2D([0], [0], marker='x', linestyle='none', color='red', markersize=5, label='Defensive Coverage (Post-throw)'),
               Line2D([0], [0], marker='*', linestyle='none', color='gold', markersize=10, label='Ball Landing')
              ]
    labels = [x.get_label() for x in handles]
    
    # plot the data
    fig, ax = plt.subplots(figsize=(10, 6)) # Added figsize so it fits nicely
    
    # filter pre-throw (input) and post-throw (output) data for specific game and play
    play_in = pre_data[(pre_data['game_id'] == game_id) & (pre_data['play_id'] == play_id)].reset_index()
    play_out = post_data[(post_data['game_id'] == game_id) & (post_data['play_id'] == play_id)].reset_index()
    
    # sequentially plot each position group
    for role in pos_map:
        # filter for position group
        dat_in = play_in[play_in['player_role'] == role]
        # save nfl_ids (for matching post-throw data to pre-throw position group)
        pos_map[role]['nfl_ids'] = list(np.unique(dat_in['nfl_id']))
        # plot each position group separately
        scatter = ax.scatter(dat_in.x, dat_in.y, c = dat_in.frame_id, cmap = pos_map[role]['color'],
                    marker = pos_map[role]['marker_pre'])
        # repeat for post-throw data, only if post-throw predictions (or actual movement) are available for someone in that position group
        dat_out = play_out[play_out['nfl_id'].isin(pos_map[role]['nfl_ids'])]
        if len(dat_out) > 0:
            ax.scatter(dat_out.x, dat_out.y, c = dat_out.frame_id, cmap = pos_map[role]['color'],
                        marker = pos_map[role]['marker_post'], s = 10)
    
    # plot ball land location
    ax.scatter(play_in.loc[0,'ball_land_x'], play_in.loc[0,'ball_land_y'], c = 'gold', marker = '*', s = 100, label = 'Ball Landing')
    
    # add custom legend outside of main plot display
    ax.legend(handles = handles, labels = labels, bbox_to_anchor=(1, 1.02), loc='upper left')
    
    # set y limits to allow a few yards for ball to land out of bounds
    ax.set_ylim(-4, 57.3)
    # save appropriate x limits
    xlim = ax.get_xlim()
    
    # add hash marks
    ax.hlines([23.58, 29.75], xmin = max(10, xlim[0]), xmax = min(110, xlim[1]), linestyle = 'dotted', colors = 'grey', alpha = 0.5, zorder = 0)
    ax.axhline(0, c = 'black', lw = 2, zorder = 0)
    ax.axhline(53.3, c = 'black', lw = 2, zorder = 0)
    
    # remove yticks and set custom xticks to reflect how yardlines are marked on a football field
    ax.set_yticks([])
    ax.set_xticks([int(x) for x in ax.get_xticks() if x % 5 == 0])
    ax.set_xticklabels(['G' if (x in [10, 110]) else 'OOB' if (x in [0, 120]) else '' if (x in [-5, 5, 115, 125]) else str(int(x-10)) if (x <= 60) else str(int(110-x)) for x in ax.get_xticks()])
    for tick in ax.get_xticks():
        if (tick > 10) & (tick < 110):
            ax.axvline(tick, c = 'grey', alpha = 0.5, zorder = 0)
        elif (tick in [5, 115]) | (tick < 0) | (tick > 120): pass
        else:
            ax.axvline(tick, c = 'black', linewidth = 2, zorder = 0)
    
    
    # move tick labels onto the field, again to make it look like a football field
    ax.tick_params(axis='x', direction='in', pad=-30, labelsize=15)
    # reset x limits
    ax.set_xlim(xlim)
    
    plt.title(f"Game ID {game_id}, Play ID {play_id}")
    
    return fig

# Plot the figurte
pre_post_figure = plot_pre_post(df_pre, df_post, game_id, play_id)
st.pyplot(pre_post_figure)
