import matplotlib.pyplot as plt

# We can hopefully use this class for all plotting related tasks with our data
class Plotter():
    def __init__(self, df):
        self.df = df

    # Plots the player's positons for a certain play
    def plotPlay(self, play):
        plt.figure(figsize=(10, 6))
        play_df = self.df[self.df['play_id'] == play].copy()

        for name, data in play_df.groupby('player_name'):
            plt.scatter(data['x'], data['y'],s=50,alpha=0.8,label=f"{name} ({data['player_role'].iloc[0]})")

        plt.xlabel('X-POS')
        plt.ylabel('Y-POS')
        plt.title(f'Player positions for play {play}')
        plt.legend()
        plt.show()