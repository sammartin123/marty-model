"""Team names and display colours (primary, alternate) tuned for a dark page."""
TEAMS = {
 'ARI': ('Arizona', 'Cardinals', '#C8334F', '#FFB612'), 'ATL': ('Atlanta', 'Falcons', '#D32F3F', '#B0B7BC'),
 'BAL': ('Baltimore', 'Ravens', '#7B5BD6', '#C9A227'), 'BUF': ('Buffalo', 'Bills', '#2E6BE6', '#E0364A'),
 'CAR': ('Carolina', 'Panthers', '#1A9BDC', '#B0B7BC'), 'CHI': ('Chicago', 'Bears', '#E5551B', '#3A5BA9'),
 'CIN': ('Cincinnati', 'Bengals', '#FF5A1F', '#C9C9C9'), 'CLE': ('Cleveland', 'Browns', '#FF4A1C', '#8B5A2B'),
 'DAL': ('Dallas', 'Cowboys', '#8FA6C8', '#2F5BB7'), 'DEN': ('Denver', 'Broncos', '#FB5A1E', '#3B6FD1'),
 'DET': ('Detroit', 'Lions', '#1F8FD6', '#B0B7BC'), 'GB': ('Green Bay', 'Packers', '#FFB612', '#3E8E5A'),
 'HOU': ('Houston', 'Texans', '#D1263F', '#4477CE'), 'IND': ('Indianapolis', 'Colts', '#3B73D9', '#C9C9C9'),
 'JAX': ('Jacksonville', 'Jaguars', '#1BA3A6', '#D7A22A'), 'KC': ('Kansas City', 'Chiefs', '#E31837', '#FFB81C'),
 'LA': ('Los Angeles', 'Rams', '#3D6FE0', '#FFD100'), 'LAC': ('Los Angeles', 'Chargers', '#2BA6E8', '#FFC20E'),
 'LV': ('Las Vegas', 'Raiders', '#B5BCC2', '#7A7F85'), 'MIA': ('Miami', 'Dolphins', '#12B5B0', '#FC6B2D'),
 'MIN': ('Minnesota', 'Vikings', '#8B5CF0', '#FFC62F'), 'NE': ('New England', 'Patriots', '#3D63C9', '#D12E43'),
 'NO': ('New Orleans', 'Saints', '#D3BC8D', '#8C8C8C'), 'NYG': ('New York', 'Giants', '#3A62D8', '#D62B45'),
 'NYJ': ('New York', 'Jets', '#2FA36B', '#C9C9C9'), 'PHI': ('Philadelphia', 'Eagles', '#1FA08A', '#A5ACAF'),
 'PIT': ('Pittsburgh', 'Steelers', '#FFB612', '#4F8BFF'), 'SEA': ('Seattle', 'Seahawks', '#69BE28', '#3D6FB8'),
 'SF': ('San Francisco', '49ers', '#D8343F', '#C9A227'), 'TB': ('Tampa Bay', 'Buccaneers', '#E0352B', '#B1A99A'),
 'TEN': ('Tennessee', 'Titans', '#5BA4E6', '#D83A4A'), 'WAS': ('Washington', 'Commanders', '#C2414B', '#FFB612'),
}

def _rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]

def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b))) ** 0.5

def colours(home, away):
    """Pick one colour per team that reads clearly against the other."""
    h, a = TEAMS[home][2], TEAMS[away][2]
    if _dist(h, a) < 150:
        a = TEAMS[away][3]
        if _dist(h, a) < 150:
            h = TEAMS[home][3]
    return {home: h.lower(), away: a.lower()}
