from auth_db import init_db, create_user, verify_user, get_user_by_username, get_user_by_email

if __name__ == '__main__':
    init_db()
    existing = get_user_by_username("testuser")
    if existing:
        print("testuser already exists")
    else:
        ok = create_user("Test User", "testuser", "test@example.com", "password123")
        print("create_user returned:", ok)
    user = verify_user("testuser", "password123")
    print("verify_user result:", bool(user))
    if user:
        print("username:", user.get('username'), "name:", user.get('name'))
