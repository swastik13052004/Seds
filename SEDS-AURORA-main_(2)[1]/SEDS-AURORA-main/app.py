import os
from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import timedelta
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Function to process the .mseed file using obspy
def process_mseed(file_path):
    # Read the mseed file
    st = read(file_path)
    tr = st[0]
    tr_times = tr.times()
    tr_data = tr.data
    starttime = tr.stats.starttime.datetime

    # STA/LTA Parameters
    sta_len = 120  # seconds
    lta_len = 600  # seconds
    cft = classic_sta_lta(tr_data, int(sta_len * tr.stats.sampling_rate), int(lta_len * tr.stats.sampling_rate))

    # Trigger thresholds
    thr_on = 4
    thr_off = 1.5
    on_off = trigger_onset(cft, thr_on, thr_off)

    # Detection times and compiling data
    detection_times = []
    for triggers in on_off:
        on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
        on_time_str = on_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
        detection_times.append(on_time_str)

    detect_df = pd.DataFrame({'filename': [os.path.basename(file_path)] * len(detection_times), 
                              'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times})

    # Plot the STA/LTA characteristic function
    fig, ax = plt.subplots()
    ax.plot(tr_times, cft, label='STA/LTA')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('STA/LTA')
    ax.legend()

    return detect_df, fig

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded .mseed file
        detection_df, fig = process_mseed(file_path)

        # Convert DataFrame to CSV
        output_csv = io.StringIO()
        detection_df.to_csv(output_csv, index=False)
        output_csv.seek(0)

        # Convert plot to PNG
        fig_io = io.BytesIO()
        FigureCanvas(fig).print_png(fig_io)
        fig_io.seek(0)

        return render_template('result.html', csv_data=output_csv.getvalue(), figure=fig_io.getvalue().decode('latin1'))

    return render_template('prediction.html')

@app.route('/download_csv')
def download_csv():
    csv_data = request.args.get('csv_data')
    output = io.StringIO(csv_data)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='Seismic_Signals_Report.csv')

if __name__ == '__main__':
    app.run(debug=True)
