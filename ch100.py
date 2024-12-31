# # %% [markdown]
# # Loading and understanding data
# # %%
import pandas as pd

# # Load dataset
# df = pd.read_csv('hot100.csv')

# # Check if it's loaded correctly
# print("Dataset Shape:", df.shape)

# # %%
# # Display dataset columns and first few rows
# print("\nColumns:", df.columns.tolist())
# print("\nFirst few rows:")
# print(df.head())

# # %%
# # Dataset info
# print("\nDataset Info:")
# df.info()

# # %%
# #Rename columns
# # Rename 'Date' column to 'weekid'
# df.rename(columns={'Date': 'weekid'}, inplace=True)
# # Rename 'Song' column to 'song'
# df.rename(columns={'Song': 'song'}, inplace=True)
# # Rename 'Artist' column to 'performer'
# df.rename(columns={'Artist': 'performer'}, inplace=True)
# # Rename 'Rank' column to 'week_position'
# df.rename(columns={'Rank': 'week_position'}, inplace=True)
# # Rename 'Peak Position' column to 'peak_position'
# df.rename(columns={'Peak Position': 'peak_position'}, inplace=True)
# # Rename 'Weeks in Charts' column to 'weeks_on_chart'
# df.rename(columns={'Weeks in Charts': 'weeks_on_chart'}, inplace=True)

# # %% [markdown]
# # Data conversion and cleaning

# # %%
# # Convert 'weekid' to datetime
# df['weekid'] = pd.to_datetime(df['weekid'])

# # Extract month and week of year from the date
# df['month'] = df['weekid'].dt.month
# df['week_of_year'] = df['weekid'].dt.isocalendar().week

# # Create a boolean feature for December
# df['is_december'] = df['month'] == 12

# # Print the head of the dataframe to see new columns
# print(df[['weekid', 'month', 'week_of_year', 'is_december']].head())

# # %%
# # Check for missing values
# print("Missing values by column:")
# print(df.isnull().sum())

# # Check for duplicate rows
# print("\nDuplicate rows:", df.duplicated().sum())

# # %%
# # Strip whitespace and capitalize the first letter of each word in 'song' and 'performer'
# df['song'] = df['song'].str.strip().str.title()
# df['performer'] = df['performer'].str.strip().str.title()

# # Print the first few rows to see changes
# print(df[['song', 'performer']].head())

# %%
# Create a clean copy and save to a new CSV file
# df_clean = df.copy()
#df_clean['weekid'] = pd.to_datetime(df_clean['weekid'])
#df_clean['year'] = df['weekid'].dt.year

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("sharkbait1223/billboard-top-100-christmas-carol-dataset")

# print("Path to dataset files:", path)

# # %%
# import os
# import pandas as pd

# # List files in the directory
# files = os.listdir(path)
# print("Files in directory:", files)

# # Assuming the CSV file is the first file in the directory
# csv_file = os.path.join(path, files[0])

# # Load dataset
# df = pd.read_csv(csv_file)


#holiday5817 = pd.read_csv('billboard_christmas_clean.csv')

# #holiday_song_titles = [
#     "Snowman",
#     "You Make It Feel Like Christmas",
#     "Santa Tell Me",
#     "Underneath the Tree",
#     "Like It's Christmas",
#     "Cozy Little Christmas",
#     "Christmas Magic",
#     "O Tannenbaum",
#     "Christmas",
#     "Sleigh",
#     "Chimney",
#     "Saint Nick",
#     "Santa Claus",
#     "Jingle",
#     "Bells",
#     "White Winter",
#     "Little Drummer Boy",
#     "2000 Miles",
#     "Santa",
#     "Silver Bells",
#     "Favorite Things",
#     "Fairytale of New York",
#     "Player's Ball",
#     "Baby, It's Cold Outside"
# ]

# # Filter songs that are in the holiday5817 dataframe
# holiday_songs_df = df[df['song'].isin(holiday5817['song'])]

# # Filter songs that contain titles and artists from the holiday_song_titles list
# holiday_songs_titles = df[df.apply(lambda row: any(title.lower() in row['song'].lower() for title in holiday_song_titles) and any(performer.lower() in row['performer'].lower() for performer in holiday5817['performer']), axis=1)]

# # Combine both filters
# holiday_songs_combined = pd.concat([holiday_songs_df, holiday_songs_titles]).drop_duplicates()

#Cleans weeks_on_chart data to make min weeks 1
# holiday_songs_combined['weeks_on_chart'] = holiday_songs_combined['weeks_on_chart'].replace('-', 1)

# print(holiday_songs_combined)
# holiday_songs_combined.to_csv('hot100_clean.csv', index=False)

# %%
df = pd.read_csv('hot100_clean.csv')
# Extract year from 'weekid' and create a new column 'year'
df['weekid'] = pd.to_datetime(df['weekid'], errors='coerce')
df['year'] = df['weekid'].dt.year
# Replace '-' in 'weeks_on_chart' with 1
df['weeks_on_chart'] = df['weeks_on_chart'].replace('-', 1)
df['weeks_on_chart'] = pd.to_numeric(df['weeks_on_chart'], errors='coerce').fillna(0)

# %% [markdown]
# **Descriptive Stats** - 
# most frequent songs, artists, most successful songs by peak position and weeks on chart

# %%
# Generate descriptive statistics for numerical columns
print("Chart Statistics:")
print(df[['week_position', 'peak_position', 'weeks_on_chart']].describe())

# %%
# Analyze the top 10 most frequent songs
print("\nTop 10 Most Frequent Songs:")
print(df['song'].value_counts().head(10))

# %%
# Identify top 10 artists by the number of appearances
print("\nTop 10 Artists by Appearances:")
print(df['performer'].value_counts().head(10))

# %%
# Group and aggregate data by song and performer to calculate success metrics
success_metrics = df.groupby(['song', 'performer']).agg({
    'peak_position': 'min',
    'weeks_on_chart': 'max',
    'year': ['min', 'max']
}).round(2)

# Display aggregated success metrics for songs and performers sorted by peak position min
print("\nMost Successful Songs:")
print(success_metrics.sort_values(('peak_position', 'min')).head())

# %% [markdown]
# **Analyzing time based patterns** - yearly, monthly, decadal

# %%
# Analyze yearly trends
yearly_stats = df.groupby('year').agg({
    'song': 'nunique',        # Number of unique songs
    'performer': 'nunique',   # Number of unique performers
    'peak_position': 'min'    # Minimum peak position (i.e., highest charted)
}).round(2)

print("Songs Per Year:")
print(yearly_stats.sort_values('song', ascending=False).head())

# %%
# Monthly patterns analysis
monthly_stats = df.groupby('month')['song'].nunique()

print("\nSongs by Month:")
print(monthly_stats.sort_values(ascending=False))

# %%
# Derive the decade from the year
df['decade'] = (df['year'] // 10) * 10

# Decadal trends analysis
decade_stats = df.groupby('decade').agg({
    'song': 'nunique',        # Number of unique songs
    'performer': 'nunique',   # Number of unique performers
    'peak_position': ['min','mean']    # Minimum peak position
})

print("\nTrends by Decade:")
print(decade_stats)

# %% [markdown]
# **Performance Metrics** - song and performer by years active, whether they reached top 10, peak position, most enduring based n total weeks, comeback songs by yeats active, and most consistent performers with a min of 5 appearances

# %%
# Group data by 'song' and 'performer'
grouped_data = df.groupby(['song', 'performer'])
print(grouped_data.first().head())  # Displaying the first entry for each group


# %%
# Calculate performance metrics using aggregation
performance_metrics = grouped_data.agg({
    'peak_position': 'min',
    'weeks_on_chart': 'max',
    'week_position': 'mean',
    'year': ['count', 'min', 'max']
}).round(2)

print(performance_metrics.head())

# %%
# Rename columns for clarity
performance_metrics.columns = [
    'best_position', 'total_weeks', 'avg_position',
    'appearances', 'first_year', 'last_year'
]

# Add derived columns 'years_active' and 'reached_top_10'
performance_metrics['years_active'] = (
    performance_metrics['last_year'] -
    performance_metrics['first_year'] + 1
)

# Calculating 'reached_top_10'
success_threshold = 10  # Define success as reaching top 10
performance_metrics['reached_top_10'] = performance_metrics['best_position'] <= success_threshold

# %%
# Displaying the most successful songs by peak position
print("Most Successful Songs (by peak position):")
print(performance_metrics.sort_values('best_position').head())

# Displaying the most enduring songs by total weeks
print("\nMost Enduring Songs (by total weeks):")
print(performance_metrics.sort_values('total_weeks', ascending=False).head())

# Displaying best comeback songs by years active
print("\nBest Comeback Songs (by years active):")
print(performance_metrics.sort_values('years_active', ascending=False).head())

# Displaying most consistent performers with at least 5 appearances
print("\nMost Consistent Performers (by average position, min 5 appearances):")
multiple_hits = performance_metrics[performance_metrics['appearances'] >= 5]
print(multiple_hits.sort_values('avg_position').head())

# %% [markdown]
# **Visualizations** 

# %%
import plotly.express as px


# %%
import os

# Aggregate data to get yearly counts of unique songs
yearly_songs = df.groupby('year')['song'].nunique().reset_index()

# Create a line chart
fig = px.line(yearly_songs, 
              x='year', 
              y='song',
              title='Christmas Songs on Billboard Hot 100 by Year')
fig.update_traces(line_color='darkgreen')


# %%
# Generate a scatter plot of all the peak positions over time
peakposition = px.scatter(df,
                 x='weekid',
                 y='peak_position',
                 color='song',
                 title='Peak Positions Over Time')

# Hide the legend
peakposition.update_layout(
    yaxis=dict(autorange="reversed"),
    showlegend=False)


# %%
#Top 25
# Generate a scatter plot of all the peak positions over time
fig = px.scatter(df[df['peak_position'] <= 25],
                x='weekid',
                y='peak_position',
                color='song',
                title='Peak Positions Over Time - Top 25')


# %%
# Find top performers
top_performers = df.groupby('performer')['song'].nunique().sort_values(ascending=False)
top_performers = top_performers[top_performers > 1]
# Create a bar chart
fig = px.bar(x=top_performers.index,
            y=top_performers.values,
            title='Top 10 Christmas Song Performers by #unique songs on Hot 100',
            color_discrete_sequence=['darkred'])

# Set y-axis ticks to integers and name y-axis
fig.update_layout(yaxis=dict(dtick=1, title='#Christmas Songs on Hot 100'))


# %%
import pandas as pd

# Read the dataset
df = pd.read_csv('hot100_clean.csv')
# Extract year from 'weekid' and create a new column 'year'
df['weekid'] = pd.to_datetime(df['weekid'], errors='coerce')
df['year'] = pd.to_datetime(df['weekid']).dt.year

# Group by year and aggregate
yearly_stats = df.groupby('year').agg({
    'song': 'nunique',
    'peak_position': 'min',
    'week_position': 'mean'
}).reset_index()

# Print prepared data
print(yearly_stats)

# %%
# Create a line chart using Plotly Express
fig = px.line(yearly_stats,
            x='year',
            y='song',
            custom_data=['peak_position', 'week_position'])

fig.update_traces(
    hovertemplate="<br>".join([
        "Year: %{x}",
        "Number of Songs: %{y}",
        "Best Position: #%{customdata[0]}",
        "Average Position: #%{customdata[1]:.1f}"
    ])
)

# Customize layout
fig.update_layout(
    title={
        'text': 'Christmas Songs on Billboard Hot 100',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title="Year",
    yaxis_title="Number of Songs",
    hovermode='x unified',
    plot_bgcolor='#D7FFE4',
    paper_bgcolor='#FFCCCB'
)


# %%
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

# Add scatter trace
df['weeks_on_chart'] = df['weeks_on_chart'].replace('-', 1)
df['weeks_on_chart'] = pd.to_numeric(df['weeks_on_chart'], errors='coerce').fillna(0)
max_weeks = max(df['weeks_on_chart']) if max(df['weeks_on_chart']) > 0 else 1
sizeref = 2 * max_weeks / (40 ** 2)

fig.add_trace(
    go.Scatter(
        x=df['weekid'],
        y=df['week_position'],
        mode='markers',
        marker=dict(
            size=df['weeks_on_chart'],
            sizemode='area',
            sizeref=sizeref,
            color=df['peak_position'],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Peak Position')
        ),
        text=[f"Song: {song}<br>Performer: {performer}"
            for song, performer in zip(df['song'], df['performer'])],
        hovertemplate="%{text}<br>"
                    "Date: %{x}<br>" +
                    "Position: %{y}<br>" +
                    "Weeks on Chart: %{marker.size}<br>" +
                    "<extra></extra>"
    )
)


fig.add_trace(
    go.Scatter(       
        hovertemplate="%{text}<br>"
                    "Date: %{x}<br>" +
                    "Position: %{y}<br>" +
                    "Weeks on Chart: %{marker.size}<br>" +
                    "<extra></extra>"
    )
)


# Update layout
fig.update_layout(
    title='Song Performance Matrix',
    xaxis_title='Date',
    yaxis_title='Chart Position',
    yaxis=dict(
        autorange="reversed",
        gridcolor='lightgray',
    ),
    xaxis=dict(gridcolor='lightgray'),
    plot_bgcolor='white',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
    ),
    
    # Static Height and Width will not resize in browser
    width=600,
    height=480
)

# Add annotations
fig.add_annotation(
    x=1980,
    y=91,
    text="Empire Strikes Back Released",
    showarrow=True,
    arrowhead=1,
    bgcolor="white"
)

fig.add_annotation(
    x=1984,
    y=65,
    text="Band Aid Release",
    showarrow=True,
    arrowhead=1,
    bgcolor="white"
)

# Add range slider
fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
)




# %% [markdown]
# **Interactive dashboard web app** 

from dash import html, dcc, Dash, callback, dash_table
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd

# Load Data
df = pd.read_csv("hot100_clean.csv")  # Ensure this file exists
df['weekid'] = pd.to_datetime(df['weekid'])
# Extract year from 'weekid' and create a new column 'year'
df['weekid'] = pd.to_datetime(df['weekid'], errors='coerce')
df['year'] = df['weekid'].dt.year

yearly_songs = df.groupby('year').agg({'song': 'nunique', 'week_position': 'mean'}).reset_index()
performer_songs = df.groupby('performer')['song'].nunique().reset_index()

# Prepare summary data
top_songs = df.groupby(['song', 'performer']).agg({
    'peak_position': 'min',
    'weeks_on_chart': 'max',
    'year': ['min', 'max']
}).reset_index()

top_songs.columns = ['Song', 'Performer', 'Peak Position', 'Weeks on Chart', 'First Year', 'Last Year']
top_songs.sort_values('Peak Position', inplace=True)

# Colors
COLORS = {
    'red': '#D42426',
    'green': '#165B33',
    'white': '#F8F9F9',
    'light_red': '#F6E7E7',
}


# Bar Chart
bar_chart = px.bar(
    performer_songs.sort_values('song', ascending=False),
    x='performer',
    y='song',
    title='Christmas Songs on Billboard by Performer',
    color_discrete_sequence=[COLORS['green']]
)

# Summary DataTable
table_df = df.groupby(['song', 'performer']).agg({
    'peak_position': 'min',
    'weeks_on_chart': 'max',
    'year': ['min', 'max']
}).reset_index()
table_df.columns = ['Song', 'Performer', 'Peak Position', 'Weeks on Chart', 'First Year', 'Last Year']

# Scatter Chart (Song Performance Matrix)
df['weeks_on_chart'] = df['weeks_on_chart'].replace('-', 1)
df['weeks_on_chart'] = pd.to_numeric(df['weeks_on_chart'], errors='coerce').fillna(0)
max_weeks = max(df['weeks_on_chart']) if max(df['weeks_on_chart']) > 0 else 1
sizeref = 2 * max_weeks / (40 ** 2)

scatter_chart = go.Figure()
scatter_chart.add_trace(
    go.Scatter(
        x=df['weekid'],
        y=df['week_position'],
        mode='markers',
        marker=dict(
            size=df['weeks_on_chart'],
            sizemode='area',
            sizeref=sizeref,
            color=df['peak_position'],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Peak Position')
        ),
        text=[
            f"Song: {song}<br>Performer: {performer}"
            for song, performer in zip(df['song'], df['performer'])
        ],
        hovertemplate="%{text}<br>Date: %{x}<br>Position: %{y}<br>Weeks on Chart: %{marker.size}<br><extra></extra>"
    )
)
scatter_chart.update_layout(
    title='Song Performance Matrix',
    xaxis_title='Date',
    yaxis_title='Chart Position',
    yaxis=dict(autorange="reversed", gridcolor='lightgray'),
    xaxis=dict(gridcolor='lightgray'),
    plot_bgcolor='white',
    hoverlabel=dict(bgcolor="white", font_size=12),
    width=600,
    height=480,
)
scatter_chart.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

# Initialize App
app = Dash(__name__)

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Christmas Songs Dashboard 🎄', style={'color': COLORS['white']}),
        html.P('Explore the magic of Christmas songs on the Billboard charts!', style={'color': COLORS['white']})
    ], style={
        'backgroundColor': COLORS['red'],
        'padding': '20px',
        'textAlign': 'center',
        'borderRadius': '10px',
        'margin': '10px'
    }),

    # Main Content
    html.Div([
        # Left Column: Bar Chart
        html.Div([
            html.H3('Performer Songs Chart', style={'color': COLORS['green']}),
            dcc.Graph(figure=bar_chart)
        ], style={
            'width': '48%',
            'display': 'inline-block',
            'padding': '20px',
            'backgroundColor': COLORS['white'],
            'borderRadius': '10px',
            'marginRight': '2%'  # Add margin to the right for spacing
        }),

        # Right Column: Scatter Chart
        html.Div([
            html.H3('Song Performance Matrix', style={'color': COLORS['green'], 'textAlign': 'center'}),
            dcc.Graph(figure=scatter_chart)
        ], style={
            'width': '48%',
            'display': 'inline-block',
            'padding': '20px',
            'backgroundColor': COLORS['white'],
            'borderRadius': '10px'
        })
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}),
    
    # Quick Stats
    html.Div([
        html.H3('Quick Stats', style={'color': COLORS['green'], 'textAlign': 'center'}),
        html.Div([
            html.P(f"Total Songs: {df['song'].nunique()}"),
            html.P(f"Total Performers: {df['performer'].nunique()}"),
            html.P(f"Years Covered: {df['year'].min()} - {df['year'].max()}"),
            html.P(f"Top Performer: {df.loc[df['peak_position'].idxmin(), 'performer']}"),
            html.P(f"Peak Position: #{df['peak_position'].min()}")
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['light_red'],
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '0 auto',
            'width': '50%'
        })
    ], style={'padding': '20px'}),

    # Searchable DataTable
    html.Div([
        dcc.Input(
            id='search-box',
            type='text',
            placeholder='Search songs or artists...',
            style={
                'width': '100%',
                'padding': '10px',
                'marginBottom': '10px',
                'borderRadius': '5px',
                'border': f'2px solid {COLORS["green"]}'
            }
        ),
        dash_table.DataTable(
            id='song-table',
            columns=[{"name": i, "id": i} for i in table_df.columns],
            data=table_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': COLORS['green'],
                'color': COLORS['white'],
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px'
            },
            style_data_conditional=[{
                'if': {'row_index': 'odd'},
                'backgroundColor': COLORS['light_red']
            }],
            page_size=10,
            sort_action='native',
            filter_action='native'
        )
    ]),
    
    #Peakpositions
    html.Div([
    html.H1('Peak Positions Over Time', style={'textAlign': 'center'}),
    dcc.Graph(
        id='peak-position-plot',
        figure=peakposition
    )
    ]),

    # Top songs, seasonal performance, artist comparison
html.Div([
    # Christmas Songs Timeline
    html.Div([
        html.H3('Christmas Songs Timeline 📊', style={'color': COLORS['green']}),
        dcc.Graph(id='timeline-chart')
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    }),

    # Top Christmas Songs
    html.Div([
        html.H3('Top Christmas Songs 🎵', style={'color': COLORS['green']}),
        dash_table.DataTable(
            id='top-songs-table',
            columns=[{"name": i, "id": i} for i in top_songs.columns],
            data=top_songs.nsmallest(10, 'Peak Position').to_dict('records'),
            style_header={'backgroundColor': COLORS['green'], 'color': 'white'},
            style_cell={'textAlign': 'left'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': COLORS['light_red']}],
            export_format='csv'
        )
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    }),

    # Seasonal Performance
    html.Div([
        html.H3('Seasonal Performance 🌟', style={'color': COLORS['green']}),
        dcc.Graph(id='seasonal-chart')
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    }),

    # Artist Comparison
    html.Div([
        html.H3('Artist Comparison 🎤', style={'color': COLORS['green']}),
        dcc.Dropdown(
            id='artist-dropdown',
            options=[{'label': artist, 'value': artist} for artist in df['performer'].unique()],
            value=[df['performer'].iloc[0]],
            multi=True,
            style={'marginBottom': '10px'}
        ),
        dcc.Graph(id='artist-chart')
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    })
    ], style={
        'padding': '20px'
    })
])

# Callback for search functionality
@app.callback(
    Output('song-table', 'data'),
    Input('search-box', 'value')
)
def update_table(search_term):
    if not search_term:
        return table_df.to_dict('records')

    filtered_df = table_df[
        table_df['Song'].str.contains(search_term, case=False, na=False) |
        table_df['Performer'].str.contains(search_term, case=False, na=False)
    ]
    return filtered_df.to_dict('records')

@callback(
    [Output('timeline-chart', 'figure'),
    Output('seasonal-chart', 'figure'),
    Output('artist-chart', 'figure')],
    [Input('artist-dropdown', 'value')]
)
def update_charts(selected_artists):
    # Timeline chart displaying number of songs per year
    yearly_data = df.groupby('year').agg({'song': 'nunique', 'peak_position': 'min'}).reset_index()
    timeline = px.line(yearly_data, x='year', y='song', title='Christmas Songs on Billboard Over Time')
    timeline.update_traces(line_color=COLORS['red'])

    # Seasonal average positioning by month
    monthly_data = df.groupby(df['weekid'].dt.month)['week_position'].mean().reset_index()
    seasonal = px.bar(monthly_data, x='weekid', y='week_position', title='Average Chart Position by Month')
    seasonal.update_traces(marker_color=COLORS['green'])
    seasonal.update_layout(yaxis_autorange='reversed')

    # Artist performance over time
    artist_data = df[df['performer'].isin(selected_artists)]
    artist_perf = px.line(artist_data, x='weekid', y='week_position', color='performer', title='Artist Performance Over Time')
    artist_perf.update_layout(yaxis_autorange='reversed')

    return timeline, seasonal, artist_perf

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)