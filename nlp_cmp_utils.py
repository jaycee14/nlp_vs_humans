"""Util functions"""

def get_config(filepath):
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            k, v = line.rstrip().split('=')
            config[k] = v

    return config


def create_db_url(config, host_addr):
    url = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=config['POSTGRES_USER'],
                                                                pw=config['POSTGRES_PASSWORD'],
                                                                url=host_addr,
                                                                db=config['POSTGRES_DB'], )

    return url


