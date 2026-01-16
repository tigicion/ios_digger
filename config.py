"""Configuration for iOS Digger."""

REGIONS = {
    "cn": "中国",
    "us": "美国",
    "jp": "日本",
    "kr": "韩国",
    "sg": "新加坡",
    "th": "泰国",
    "vn": "越南",
    "id": "印尼",
}

CATEGORIES = {
    "效率": 6007,
    "健康健身": 6013,
    "生活": 6012,
    "工具": 6002,
    "社交": 6005,
    "财务": 6015,
    "教育": 6017,
    "娱乐": 6016,
    "摄影与录像": 6008,
    "音乐": 6011,
}

DEFAULT_REGIONS = ["cn", "us"]
DEFAULT_LEVEL = "standard"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_TOP = 10

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_RSS_URL = "https://itunes.apple.com/{region}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
ITUNES_CATEGORY_RSS_URL = "https://itunes.apple.com/{region}/rss/topfreeapplications/genre={genre_id}/limit={limit}/json"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
