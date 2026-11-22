# -*- coding: utf-8 -*-
"""问询函重提取 v2：从公告推断"函件事件"（原始函+回复证据+核查意见证据去重），
口径=仅问询函（关注函已排除）。先用舞弊公司（有人工清单=真值）做校准。"""
import urllib.request, urllib.parse, json, ssl, time, re, sys
import pandas as pd

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
     'X-Requested-With': 'XMLHttpRequest', 'Referer': 'http://www.cninfo.com.cn/'}

def req(url, data):
    body = urllib.parse.urlencode(data).encode()
    r = urllib.request.Request(url, data=body, headers=H, method='POST')
    return json.loads(urllib.request.urlopen(r, timeout=20, context=ctx).read())

def pull_all(code):
    col = 'sse' if code.startswith('6') else 'szse'
    ts = req('http://www.cninfo.com.cn/new/information/topSearch/query', {'keyWord': code, 'maxNum': '10'})
    org = next((x['orgId'] for x in ts if x['code'] == code), None)
    if not org: return []
    out = []
    for page in range(1, 6):
        r = req('http://www.cninfo.com.cn/new/hisAnnouncement/query', {
            'pageNum': page, 'pageSize': 50, 'column': col, 'tabName': 'fulltext', 'plate': '',
            'stock': f'{code},{org}', 'searchkey': '问询函', 'secid': '', 'category': '', 'trade': '',
            'seDate': '2014-01-01~2023-12-31', 'sortName': '', 'sortType': '', 'isHLtitle': 'true'})
        anns = r.get('announcements') or []
        for a in anns:
            t = a.get('announcementTitle', '').replace('<em>', '').replace('</em>', '')
            tm = a.get('announcementTime', 0)
            y = int(time.strftime('%Y', time.localtime(tm / 1000))) if tm else 0
            md = time.strftime('%Y-%m-%d', time.localtime(tm / 1000)) if tm else ''
            out.append((y, md, t))
        if len(anns) < 50: break
        time.sleep(0.2)
    return out

NOISE = ['延期回复', '延期公告', '修订稿', '募集说明书', '提示性公告', '反馈意见', '落实函']
REPLY = ['回复', '回函', '答复', '核查意见', '专项说明', '独立意见', '专项核查']

def letter_type(t):
    m = re.search(r'(20\d\d)\s*年?\s*年度?报告', t)
    if ('年报' in t or '年度报告' in t) and m: return f'年报{m.group(1)}'
    if '半年报' in t or '中期报告' in t: return '半年报'
    if '季报' in t or '季度报告' in t: return '季报'
    if '业绩预告' in t: return '业绩预告'
    if '重组' in t or '重大资产重组' in t: return '重组'
    if '募集' in t or '募投' in t or '可转债' in t: return '募投'
    if '出售' in t or '转让' in t or '收购' in t or '资产' in t or '股权' in t: return '资产交易'
    if '问询函' in t: return '其他问询'
    return None

def infer_letters(anns):
    """从公告列表推断函件事件，去重键=(类型, 目标年份近似)"""
    letters = {}
    for y, md, t in anns:
        lt = letter_type(t)
        if not lt: continue
        is_noise = any(k in t for k in NOISE)
        is_reply = any(k in t for k in REPLY) and '收到' not in t
        if is_noise and not is_reply:  # 延期回复等不产生新函
            continue
        # 发函年：年报函以回复/收到公告年近似；标题带年份的用标题年
        m = re.match(r'年报(20\d\d)', lt)
        key = (lt, m.group(1) if m else y)
        # 同一函多份公告（收到+回复+核查意见）取最早日期
        if key not in letters or md < letters[key][0]:
            letters[key] = (md, t)
    return letters

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'fraud_szse'
    if which == 'fraud_szse':
        cos = {'300208': '青岛中程', '300376': '易事特', '300742': '越博动力', '002176': '江特电机'}
    else:
        cos = json.loads(open('问询重提取_目标.json').read())
    cl = pd.read_csv('问询函核对清单.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    cl['年份'] = pd.to_numeric(cl['年份'], errors='coerce')
    cl['涉财'] = pd.to_numeric(cl['涉财'], errors='coerce').fillna(0)

    for code, name in cos.items():
        anns = pull_all(code)
        letters = infer_letters(anns)
        mac = {}
        for (lt, abt), (md, t) in letters.items():
            yr = int(md[:4]) if md else abt
            mac[yr] = mac.get(yr, 0) + 1
        truth = {}
        sub = cl[(cl['firm_code'] == code) & (cl['涉财'] == 1)]
        for _, r in sub.iterrows():
            truth[int(r['年份'])] = truth.get(int(r['年份']), 0) + 1
        yrs = sorted(set(list(mac) + list(truth)))
        print(f"\n=== {name}({code}) ===")
        print(f"{'年份':6s}{'机器':>5s}{'清单':>5s}")
        cum_m = cum_t = 0
        for yy in yrs:
            cum_m += mac.get(yy, 0); cum_t += truth.get(yy, 0)
            flag = ' ✓' if mac.get(yy, 0) == truth.get(yy, 0) else ' ⚠'
            print(f"{yy:<6d}{mac.get(yy,0):>5d}{truth.get(yy,0):>5d}{flag}   (累计 {cum_m}/{cum_t})")
        time.sleep(0.3)

if __name__ == '__main__':
    main()
