"""Exercise a running local Streamlit app and capture actual browser screenshots."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]


def run_browser(args,output):
    checks=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=args.chromium_executable,
            args=['--no-sandbox','--disable-dev-shm-usage','--no-zygote','--disable-gpu','--single-process'])
        context=browser.new_context(viewport={'width':1440,'height':1800},device_scale_factor=1)
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        response=page.goto(args.url,wait_until='domcontentloaded')
        assert response and response.status==200
        expect(page.get_by_role('heading',name='Explore a symptom pattern.')).to_be_visible(timeout=45000)
        page.get_by_role('button',name='Explore model prediction').click()
        expect(page.get_by_text('Select at least one symptom to explore a prediction.',exact=True)).to_be_visible()
        checks.append('Empty input rejected in live browser')
        multiselect=page.get_by_role('combobox',name='Symptoms present')
        multiselect.click()
        page.get_by_role('option',name='Cough',exact=True).click()
        multiselect.click()
        page.get_by_role('option',name='Difficulty breathing',exact=True).click()
        multiselect.press('Escape')
        page.get_by_role('heading',name='01  Choose symptoms').click()
        page.get_by_text('I have reviewed all four symptoms; unselected means absent.',exact=True).click()
        expect(page.get_by_role('checkbox',name='I have reviewed all four symptoms; unselected means absent.')).to_be_checked()
        page.get_by_role('button',name='Explore model prediction').click()
        expect(page.locator('.result-title')).to_be_visible()
        checks.append('Symptom selection and model result rendered')
        assert 'not medical diagnosis' in page.inner_text('body')
        page.screenshot(path=str(output/'symptom_explorer.png'),full_page=True)
        page.get_by_text('Condition reference',exact=True).click()
        expect(page.get_by_role('heading',name='Inspect the source material.')).to_be_visible()
        page.get_by_text('Show unverified source entries for study',exact=True).click()
        expect(page.get_by_role('checkbox',name='Show unverified source entries for study')).to_be_checked()
        expect(page.get_by_text('• Albuterol (Rescue Inhaler)',exact=True)).to_be_visible()
        assert '110mcg' not in page.inner_text('body') and '2 puffs' not in page.inner_text('body')
        page.screenshot(path=str(output/'condition_reference.png'),full_page=True)
        checks.append('Opt-in exact source names displayed without dose instructions')
        page.get_by_text('Model & data',exact=True).click()
        expect(page.get_by_role('heading',name='Understand the model.')).to_be_visible()
        expect(page.get_by_text('Held-out accuracy',exact=True)).to_be_visible()
        expect(page.locator('[data-testid="stImage"] img').first).to_be_visible()
        page.screenshot(path=str(output/'model_evaluation.png'),full_page=True)
        checks.append('Evaluation page and plot rendered')
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        checks.append('No desktop horizontal overflow')
        # New browser context avoids carrying sidebar state into the mobile view.
        mobile=browser.new_context(viewport={'width':390,'height':1800},device_scale_factor=1,is_mobile=True,has_touch=True)
        mp=mobile.new_page();mp.on('pageerror',lambda error:errors.append(str(error)))
        mp.goto(args.url,wait_until='domcontentloaded')
        expect(mp.get_by_role('heading',name='Explore a symptom pattern.')).to_be_visible(timeout=45000)
        expect(mp.locator('[data-testid="stSidebar"]')).to_have_attribute('aria-expanded','false')
        assert mp.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        mp.screenshot(path=str(output/'mobile_explorer.png'),full_page=True)
        checks.append('390px mobile layout renders without horizontal overflow')
        assert not errors,errors
        record={'url':args.url,'browser':'Chromium','checks_passed':checks,'javascript_errors':errors,
            'desktop_viewport':[1440,1800],'mobile_viewport':[390,1800],
            'screenshots':['symptom_explorer.png','condition_reference.png','model_evaluation.png','mobile_explorer.png']}
        (output/'browser_checks.json').write_text(json.dumps(record,indent=2)+'\n')
        browser.close()
        print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    parser.add_argument('--chromium-executable',default=None)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'screenshots',help='Directory for this run; use a separate folder for hosted-app verification')
    parser.add_argument('--start-server',action='store_true',help='Start and stop the local app inside this test process')
    args=parser.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    server=None
    log=None
    try:
        if args.start_server:
            if args.url!='http://127.0.0.1:8501': parser.error('--start-server uses the default localhost URL')
            log=(output/'server.log').open('w')
            server=subprocess.Popen([sys.executable,'-m','streamlit','run','app.py','--server.port','8501','--server.address','127.0.0.1'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(100):
                if server.poll() is not None: raise RuntimeError('Streamlit exited; inspect screenshots/server.log')
                try:
                    with urllib.request.urlopen(args.url+'/_stcore/health',timeout=.5) as response:
                        if response.status==200:break
                except OSError: time.sleep(.2)
            else: raise RuntimeError('Streamlit health endpoint did not become ready')
        run_browser(args,output)
    finally:
        if server is not None:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        if log:log.close()


if __name__=='__main__':main()
