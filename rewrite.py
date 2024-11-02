# Example callback snippet:
import datetime

def parse_git_date(d):
    return datetime.datetime.strptime(d, "%a %b %d %H:%M:%S %Y %z")

def to_git_date(dt):
    return dt.strftime("%a %b %d %H:%M:%S %Y %z")

decoded_author    = commit.author_date.decode("utf-8")
decoded_committer = commit.committer_date.decode("utf-8")

try:
    dt_author    = parse_git_date(decoded_author)
    dt_committer = parse_git_date(decoded_committer)

    # If either year is >= 2025, change it to 2024
    if dt_author.year >= 2025:
        dt_author = dt_author.replace(year=2024)
    if dt_committer.year >= 2025:
        dt_committer = dt_committer.replace(year=2024)

    # Write new date back
    commit.author_date    = to_git_date(dt_author).encode("utf-8")
    commit.committer_date = to_git_date(dt_committer).encode("utf-8")

except ValueError:
    pass

