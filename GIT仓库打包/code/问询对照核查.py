# -*- coding: utf-8 -*-
"""对照公司问询三向交叉核查：主表现值 vs cninfo机器重提取 vs 深交所官方list(主板)
判定规则：现值≈机器数 → 一致；现值>机器数 → 现值偏高(可能混入关注函/回复,按新规应降)；
现值<机器数 → 现值偏低(可能漏计)；SSE公司机器覆盖薄 → 一律标注待人工复核"""
import urllib.request, urllib.parse, json, ssl, time, re
import pandas as pd

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
     'X-Requested-With': 'XMLHttpRequest', 'Referer': 'http://www.cninfo.com.cn/'}

def req(url, data):
    body = urllib.parse.urlencode(data).encode()
    r = urllib.request.Request(url, data=body, headers=H, method='POST')
    return json.loads(urllib.request.urlopen(r, timeout=20, context=ctx).read())

def pull_cninfo(code):
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
            md = time.strftime('%Y-%m-%d', time.localtime(tm / 1000)) if tm else ''
            out.append((md, t))
        if len(anns) < 50: break
        time.sleep(0.15)
    return out

NOISE = ['延期回复', '延期公告', '修订稿', '募集说明书', '提示性公告']
REPLY = ['回复', '回函', '答复', '核查意见', '专项说明', '独立意见']

def letter_type(t):
    m = re.search(r'(20\d\d)\s*年?\s*年度?报告', t)
    if ('年报' in t or '年度报告' in t) and m: return f'年报{m.group(1)}'
    if '半年报' in t or '中期报告' in t: return '半年报'
    if '季报' in t or '季度报告' in t: return '季报'
    if '业绩预告' in t: return '业绩预告'
    if '重组' in t: return '重组'
    if '募集' in t or '募投' in t or '可转债' in t: return '募投'
    if '出售' in t or '转让' in t or '收购' in t or '资产' in t or '股权' in t: return '资产交易'
    if '问询函' in t: return '其他问询'
    return None

def infer_letters(anns):
    letters = {}
    for md, t in anns:
        lt = letter_type(t)
        if not lt: continue
        is_noise = any(k in t for k in NOISE)
        is_reply = any(k in t for k in REPLY) and '收到' not in t
        if is_noise and not is_reply: continue
        m = re.match(r'年报(20\d\d)', lt)
        key = (lt, m.group(1) if m else (md[:4] if md else ''))
        if key not in letters or (md and md < letters[key][0]):
            letters[key] = (md, t)
    return letters

def main():
    m = pd.read_csv('data/数据表_全部.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    m.columns = [c.split('（')[0] for c in m.columns]
    ctrl = m[m['label'] == '0'][['firm_code', 'firm_name', 'year', 'inquiry_count_hist']].copy()
    ctrl['inquiry_count_hist'] = pd.to_numeric(ctrl['inquiry_count_hist'], errors='coerce')
    ctrl['year'] = pd.to_numeric(ctrl['year'], errors='coerce')

    try:
        off = pd.read_excel('szse_wxhj_tab2.xlsx', dtype=str)
        off['发函日期'] = pd.to_datetime(off['发函日期'], errors='coerce')
        off = off[off['函件类别'].str.contains('问询', na=False)]
    except Exception:
        off = pd.DataFrame()

    rows = []
    for code, grp in ctrl.groupby('firm_code'):
        name = grp['firm_name'].iloc[0]
        anns = pull_cninfo(code)
        letters = infer_letters(anns)
        mac_by_year = {}
        for (lt, abt), (md, t) in letters.items():
            yr = int(md[:4]) if md else None
            if yr: mac_by_year[yr] = mac_by_year.get(yr, 0) + 1
        off_sub = off[off['公司代码'] == code] if len(off) else pd.DataFrame()
        is_sse = code.startswith('6')
        for _, r in grp.iterrows():
            T = int(r['year']); cur = r['inquiry_count_hist']
            mac_cum = sum(v for y, v in mac_by_year.items() if y <= T)
            off_cum = len(off_sub[off_sub['发函日期'].dt.year <= T]) if len(off_sub) else None
            if is_sse:
                verdict = '待人工(SSE机器覆盖薄)'
            elif off_cum is not None and cur == off_cum == mac_cum:
                verdict = '三向一致'
            elif off_cum is not None and cur == off_cum:
                verdict = '现值=官方(机器漏)'
            elif cur == mac_cum:
                verdict = '现值=机器'
            elif off_cum is not None and cur > max(mac_cum, off_cum):
                verdict = '现值偏高(疑混入关注函/回复)'
            elif cur < max(mac_cum, off_cum or 0):
                verdict = '现值偏低(疑漏计)'
            else:
                verdict = '不一致待人工'
            rows.append({'firm_code': code, 'firm_name': name, 'year': T, '现值': cur,
                         'cninfo机器累计': mac_cum, '深交所官方累计': off_cum, '判定': verdict})
        print(f"{name}: 样本年{len(grp)}个 机器识别{len(letters)}封函", flush=True)
        time.sleep(0.3)

    out = pd.DataFrame(rows)
    out.to_csv('问询对照核查_结果.csv', index=False, encoding='utf-8-sig')
    print('\n=== 判定分布 ===')
    print(out['判定'].value_counts().to_string())
    print('\n=== 非一致明细 ===')
    bad = out[out['判定'] != '三向一致']
    print(bad.to_string(index=False) if len(bad) else '全部一致')

if __name__ == '__main__':
    main()
