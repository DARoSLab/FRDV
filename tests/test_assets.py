from pathlib import Path

from frdv import assets


def test_model_revisions_have_independent_caches(monkeypatch,tmp_path):
    monkeypatch.setenv('FRDV_CACHE',str(tmp_path))
    legacy=tmp_path/'mujoco_menagerie/unitree_g1/g1.xml'
    legacy.parent.mkdir(parents=True); legacy.write_text('old unversioned model')
    urls=[]
    def get(url,destination):
        urls.append(url)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="body.stl"/></asset></mujoco>'
                               if destination.suffix=='.xml' else url)
    monkeypatch.setattr(assets,'_get',get)
    monkeypatch.setenv('FRDV_MENAGERIE_REF','revision-A')
    a=Path(assets.menagerie('unitree_g1',quiet=True))
    monkeypatch.setenv('FRDV_MENAGERIE_REF','feature/revision-B')
    b=Path(assets.menagerie('unitree_g1',quiet=True))
    assert a!=b
    assert 'revision-A' in (a.parent/'meshes/body.stl').read_text()
    assert 'feature/revision-B' in (b.parent/'meshes/body.stl').read_text()
    assert len(urls)==4
    assert assets.menagerie('unitree_g1',quiet=True)==str(b)
    assert len(urls)==4
    assert legacy.read_text()=='old unversioned model'
