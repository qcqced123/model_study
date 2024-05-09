from youtube_transcript_api import YouTubeTranscriptApi
from language_setting import LANGUAGE


def youtube_script_api(url_path: str) -> str:
    """ extract the script from the given YouTube video URL

    Args:
        url_path (str): The YouTube video URL for extracting the script
    """
    output = ''
    try:
        video_id = url_path.replace('https://www.youtube.com/watch?v=', '')
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=LANGUAGE)
        for x in transcript:
            sentence = x['text']
            output += f'{sentence}\n'

    except Exception as e:
        print(e)

    return output

