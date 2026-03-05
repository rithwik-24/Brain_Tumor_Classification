from auth_db import init_db, create_user, get_user_by_username

users = [
    ("Rithwik", "rithwik24", "rithwik24@example.com", "Grithwik@24"),
    ("Alice Test", "alice", "alice@example.com", "AlicePass1"),
    ("Bob Test", "bob", "bob@example.com", "BobPass1"),
]

if __name__ == '__main__':
    init_db()
    for name, username, email, pwd in users:
        existing = get_user_by_username(username)
        if existing:
            print(f"{username} already exists")
        else:
            ok = create_user(name, username, email, pwd)
            print(f"created {username}: {ok}")
