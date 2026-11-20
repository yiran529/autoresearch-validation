"""Explicitly archive only pre-worker EADDRINUSE attempts after the screen finishes."""
from pathlib import Path
from glval.common import read_json,write_json,file_hash

HERE=Path(__file__).resolve().parent

def main():
    status=read_json(HERE/'status.json')
    assert status['status']=='finished'
    assert len(status['completed'])==32
    assert not (HERE/'retries.json').exists()
    write_json(HERE/'status-original32.json',status)
    attempts=[]
    for case in read_json(HERE/'cases.json')['cases']:
        output=Path(case['output']);process=read_json(output/'process.json')
        if process['returncode']==0:continue
        assert process['returncode']==1
        assert 'EADDRINUSE' in (output/'stderr.log').read_text()
        assert not (output/'result.json').exists()
        assert not list(output.glob('binding-rank-*.json'))
        archive=output.with_name('output-startup-eaddrinuse-attempt1')
        assert not archive.exists()
        output.rename(archive)
        attempts.append({'case':case['id'],'archive':str(archive),'process':process,
                         'stderr_hash':file_hash(archive/'stderr.log'),
                         'scope':'TCPStore listener failed before workers; no model execution',
                         'retry':'Explicit attempt2 with master port outside ephemeral range'})
    assert attempts
    write_json(HERE/'retries.json',{'attempts':attempts})
    print(','.join(r['case'] for r in attempts))

if __name__=='__main__':main()
