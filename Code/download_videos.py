from __future__ import unicode_literals
import youtube_dl
from collections import OrderedDict
import utils, config

MSRVTT_DIR = '../Data/MSR-VTT/'
MSRVTT_JSON_DATA_PATH = '../Data/MSR-VTT/train_2017/videodatainfo_2017.json'
MSRVTT_TOTAL_VIDS = 10000

def get_vid(videos, vid_id):
	for vid in videos:
		if(vid['video_id']==vid_id):
			return vid

def get_youtube_url(url):
	ydl_options = {
		'outtmpl': '%(id)s%(ext)s',
		'format':'bestvideo[height<=480]/best[height<=480]',
	}
	ydl = youtube_dl.YoutubeDL(ydl_options)
	result = None
	try:
		with ydl:
			result = ydl.extract_info(url, download=False) # We just want to extract the info )
		if not result:
			print("Failed for : "+url)
			return None, "Unknown"
		video = result
		if 'entries' in result:
			video = result['entries'][0]
		return video['url'], "Success"
	except Exception as e:
		print("Failed for : "+url)
		return None, repr(e).encode('utf-8')

def map_url_with_ids(videos):
	vid_ids = []
	vid_urls = []
	url_ids_map = OrderedDict()
	for vid in videos:
		vid_url = vid['url']
		vid_id = vid['video_id']
		if vid_url in url_ids_map:
			url_ids_map[vid_url].append(vid_id)
		else:
			url_ids_map[vid_url] = [vid_id]
		vid_ids.append(vid_id)
		vid_urls.append(vid_url)
	assert len(set(vid_ids))==MSRVTT_TOTAL_VIDS
	print "urls#:",len(set(vid_urls)),'/',len(vid_urls)
	url_ydl_map = OrderedDict()
	count = 0
	success = 0
	fail = 0
	for url in url_ids_map:
		ydl_url, status = get_youtube_url(url)
		url_ydl_map[url] = {
			"ydl_url": ydl_url,
			"status": status
		}
		if status=="Success":
			success += 1
		else:
			fail += 1
		count = count + 1
	print success,"/",count," ",fail,"/",count
	url_ydl_map["#success"] = success
	url_ydl_map["#fail"] = fail
	url_ydl_map["#count"] = count
	utils.write_to_json(url_ids_map, MSRVTT_DIR+"urls_vidids_map.json")
	utils.write_to_json(url_ydl_map, MSRVTT_DIR+"urls_ydl_map.json")

if __name__ == '__main__':
	print('loading json data...')
	data = utils.read_from_json(MSRVTT_JSON_DATA_PATH)
	videos = data['videos']
	assert len(videos)==MSRVTT_TOTAL_VIDS
	map_url_with_ids(videos)