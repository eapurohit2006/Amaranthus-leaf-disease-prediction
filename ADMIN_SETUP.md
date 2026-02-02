# Admin Account Setup

## Default Admin Credentials

The application automatically creates a default admin user when the backend starts:

- **Username/Email**: `admin@example.com`
- **Password**: `Admin@123`
- **Role**: `admin`

## Features Implemented

### ✅ Admin Account Creation
- Default admin user is automatically created on backend startup
- Admin user is stored in `users.db.json`
- Admin cannot be created through normal signup page (signup always creates "user" role)

### ✅ Role-Based Redirects
- **Admin login** → Redirects to `/admin-dashboard`
- **User login** → Redirects to `/` (home)
- **Login failure** → Shows error message

### ✅ Route Protection
- **Protected Routes**: All pages except `/login` and `/signup` require authentication
- **Admin-Only Routes**: `/admin-dashboard` is only accessible to users with `role: "admin"`
- Unauthenticated users are redirected to `/login`
- Non-admin users trying to access admin routes are redirected to home

### ✅ Logout Functionality
- Logout button clears all session data:
  - `token`
  - `username`
  - `role`
- Redirects to `/login` page after logout

## Admin Dashboard Features

The admin dashboard (`/admin-dashboard`) includes:

- **Statistics Cards**:
  - Total Users
  - Total Predictions
  - Active Today

- **Recent Predictions Table**:
  - Date/Time
  - Disease detected
  - Confidence level
  - Status (Healthy/Disease)

- **Quick Actions**:
  - View User Dashboard
  - Test Prediction
  - View All History

## Testing Admin Access

1. Start the backend server:
   ```powershell
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. You should see a message in the console:
   ```
   ✅ Default admin user created: admin@example.com / Admin@123
   ```

3. Start the frontend:
   ```powershell
   cd frontend-react
   npm run dev
   ```

4. Login with admin credentials:
   - Username: `admin@example.com`
   - Password: `Admin@123`

5. You should be redirected to `/admin-dashboard`

## API Endpoints

### Authentication
- `POST /auth/login` - Returns `{access_token, token_type, role}`
- `POST /auth/signup` - Returns `{access_token, token_type, role}` (always "user")
- `GET /auth/me` - Get current user info (requires authentication)

### Protected Endpoints
- `POST /predict` - Requires authentication (any role)

## Security Notes

- Admin users cannot be created through the signup endpoint
- All protected routes verify authentication via JWT token
- Admin routes verify role via token payload
- Tokens expire after 120 minutes (configurable in `backend/auth_simple.py`)

## User Roles

- **admin**: Full access including admin dashboard
- **user**: Standard user access (home, predict, history, results)






