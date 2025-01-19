# app.py

from flask import Flask
from controllers.main_controller import main_blueprint

def create_app():
    app = Flask(__name__)
    # Register the main blueprint where all routes are defined
    app.register_blueprint(main_blueprint)
    return app

if __name__ == '__main__':
    # Create the Flask application
    app = create_app()
    # Run the app on debug mode, port 5001
    app.run(debug=True, host="0.0.0.0",port=5001)
