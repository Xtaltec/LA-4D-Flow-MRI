"""
This script deletes all existing Paraview sources and resets the current session.
"""


from paraview.simple import *


def delete_all():
    """Delete all existing Paraview sources."""
    sources = GetSources()
    while sources:
        try:
            Delete(list(sources.values())[0])
        except:
            pass
        sources = GetSources()


def reset_session():
    """Reset the current Paraview session."""
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()


delete_all()
reset_session()
