from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, current_user, login_required, logout_user
import functools
import time
from datetime import datetime

from tqdm.notebook import tqdm
from tensorflow import keras
import datetime as dt
import re
from flask_session import Session
from sqlalchemy import create_engine
import sqlalchemy as db
import pandas as pd
from pytz import timezone
import pytz
import json

engine = db.create_engine('sqlite:///db.tide_work')

auth = Blueprint('auth', __name__)
views = Blueprint('views', __name__)


@auth.route('/', methods=['GET', 'POST'])
def login():
    data = {}
    # data.update({'success': 1, 'msg': 'Account! Login Successfully'})
    if session.get("email"):
        return redirect(url_for('views.home'))
    else:
        email = request.form.get('email')
        password = request.form.get('password')
        print(email)
        sql = "SELECT * FROM users where user_email== ? and password== ?"
        results = engine.execute(sql, (email, password)).fetchall()
        df = pd.DataFrame(results)
        if request.method == 'POST':
            if len(df) == 1:
                session["email"] = request.form.get("email")
                session['is_login'] = True
                # flash('Logged in successfully!', category='success')
                return redirect(url_for('views.home'))
            else:
                # flash('Incorrect Email and Password, try again.', category='error')
                data.update({'success': 0, 'msg': 'Incorrect Email and Password, try again.'})
    return render_template("login.html", dataset=data, is_login=True)


@auth.route('/signup', methods=['GET','POST'])
def signup():
    data = {}
    if request.method == 'POST':
        data.update({'success': 0, 'msg': 'Error! Users already exist'})
        email = request.form.get('email')
        print(email)
        sql = "SELECT * FROM 'users' where user_email == ?"
        results = engine.execute(sql, email).fetchall()
        try:
            df = pd.DataFrame(results)
            df.columns = results[0].keys()
            flash('Error! Users already exist!', category='error')

        except:
            email = request.form.get('email')
            password = request.form.get('password')
            repassword = request.form.get('re_password')
            name = request.form.get('username')
            if(password==repassword):
                print(email,password,name)
                is_admin = False
                matchs = "([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
                if (re.match(matchs, email)):
                    date_format = '%m/%d/%Y %H:%M:%S %Z'
                    date = datetime.now(tz=pytz.utc)
                    date = date.astimezone(timezone('US/Pacific'))

                    # creation_date = request.form.get('creation_date')

                    sql = """insert into users(user_email,name,password,is_admin,creation_date) VALUES (?,?,?,?,?);"""
                    engine.execute(sql, (email, name, password, is_admin, date.strftime(date_format)))
                    flash('Account! Created Successfully!', category='sucess')
            else:
                flash('Password must be match!', category='error')
        return render_template("signup.html",data=jsonify(data))
    return render_template("signup.html")


@auth.route('/logout')
def logout():
    session["email"] = None
    return redirect(url_for('auth.login'))
