from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, get_jwt
)
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from datetime import timedelta

app = Flask(__name__)

#Method not allowed oznacza, że API działa, ale nie jest używany

# CONFIG
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['JWT_SECRET_KEY'] = 'jwt-secret'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# MAIL SMTP
app.config['MAIL_SERVER'] = 'smtp.kondexor.pl'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kondzio3135@kondexor.pl'
app.config['MAIL_PASSWORD'] = 'qwerty'

db = SQLAlchemy(app)
jwt = JWTManager(app)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# BLACKLIST (prosta w pamięci)
blacklist = set()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload["jti"] in blacklist

# MODEL
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

# 📝 REJESTRACJA
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json

    if User.query.filter_by(username=data['username']).first():
        return jsonify({"msg": "Username exists"}), 400

    hashed = generate_password_hash(data['password'])
    user = User(username=data['username'], email=data['email'], password=hashed)

    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "User created"}), 201

# 🔐 LOGOWANIE
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()

    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({"msg": "Bad credentials"}), 401

    token = create_access_token(identity=user.id)
    return jsonify(access_token=token)

# 🚪 WYLOGOWANIE
@app.route('/api/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    blacklist.add(jti)
    return jsonify({"msg": "Logged out"})

# 👤 PROFIL
@app.route('/api/me', methods=['GET'])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    return jsonify({
        "username": user.username,
        "email": user.email
    })

# ⚙️ USTAWIENIA
@app.route('/api/settings', methods=['PUT'])
@jwt_required()
def settings():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    data = request.json

    if "email" in data:
        user.email = data["email"]

    if "password" in data:
        user.password = generate_password_hash(data["password"])

    db.session.commit()
    return jsonify({"msg": "Updated"})

# ❌ USUWANIE KONTA
@app.route('/api/delete', methods=['DELETE'])
@jwt_required()
def delete_account():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    db.session.delete(user)
    db.session.commit()

    return jsonify({"msg": "Account deleted"})

# 📧 RESET HASŁA - REQUEST
@app.route('/api/reset_password', methods=['POST'])
def reset_request():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()

    if user:
        token = serializer.dumps(user.email, salt='reset')
        link = f"http://localhost:5000/api/reset_password/{token}"

        msg = Message('Reset hasła',
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[user.email])
        msg.body = f'Kliknij link: {link}'
        mail.send(msg)

    return jsonify({"msg": "If email exists, message sent"})

# 🔑 RESET HASŁA - TOKEN
@app.route('/api/reset_password/<token>', methods=['POST'])
def reset_token(token):
    try:
        email = serializer.loads(token, salt='reset', max_age=3600)
    except:
        return jsonify({"msg": "Token expired"}), 400

    user = User.query.filter_by(email=email).first()
    data = request.json

    user.password = generate_password_hash(data['password'])
    db.session.commit()

    return jsonify({"msg": "Password changed"})

# 📨 PRZYPOMNIENIE LOGINU
@app.route('/api/remind_username', methods=['POST'])
def remind_username():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()

    if user:
        msg = Message('Twój login',
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[user.email])
        msg.body = f'Login: {user.username}'
        mail.send(msg)

    return jsonify({"msg": "If email exists, message sent"})

# 🚀 START
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)