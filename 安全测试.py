print 'get_price, daily', get_price('000001.XSHE')[-2:]

print  'get_price minute', get_price('000001.XSHE', '2015-12-01', '2015-12-31', frequency='minute')[-2:]

print 'get_fundamentals', get_fundamentals(query(valuation))[-2:]

write_file('test123.txt', '1234')
print 'write_file/read_file', read_file("test123.txt")


import jqdata
# print 'jqdata, get_money_flow', jqdata.get_money_flow('000001.XSHE','2015-12-25','2015-12-30', fields="change_pct")[-2:]

import os

import contextlib

@contextlib.contextmanager
def ignore_error(msg):
    try:
        yield
    except Exception as e:
        log.error(msg, 'ERROR', e)
    else:
        log.info(msg, 'OK')
    pass


with ignore_error('os.getcwd'):
    print os.getcwd()

with ignore_error("list ."):
    print os.listdir('.')

for dir in ('/', '/tmp', '/bin', '/usr/bin', '/dev/shm'):
    with ignore_error('list '+dir):
        print os.listdir(dir)

with ignore_error('write /tmp/test.txt'):
    with open('/tmp/test.txt', 'w') as f:
        f.write('test')

with ignore_error('write ./test.txt'):
    with open('test.txt', 'w') as f:
        f.write('test')

with ignore_error('read test.txt'):
    print open('test.txt').read()


def run(cmd):
    with ignore_error('run %r' % cmd):
        import commands
        print commands.getstatusoutput(cmd)


run('/usr/bin/id')
run('/bin/echo 1234')
run('/bin/cat /etc/hosts')
run('/bin/ping -c 1 www.baidu.com')
run('/bin/ping -c 1 115.239.211.112')

with ignore_error('read /etc/hosts'):
    print open("/etc/hosts").read()
