"""
This provides several utils functions for accessing the BioModels API as described on https://www.ebi.ac.uk/biomodels/docs/

See Also:
    The biomodels_api code and documentation is copied (and slightly adapated) from the basico repository that is maintained by the COPASI group and under Artistic License 2.0.
    The original code can be found at https://github.com/copasi/basico/blob/master/basico/biomodels.py
    
Note:
    The only (minor) modification with respect to original code is that we deal with special characters in :func:`~sbmltoodejax.biomodels_api.get_content_for_model` using ``urllib.parse.quote``


Examples:
    .. code-block:: python
    
        # get info for a specific model
        info = get_model_info(12)
        print(info['name'], info['files']['main'][0]['name'])
        # get all files for one model
        files = get_files_for_model(12)
        print(files['main'][0]['name'])
        # get content of specific model
        sbml = get_content_for_model(12)
        print(sbml)
        # search for model
        models = search_for_model('repressilator')
        for model in models:
           print(model['id'], model['name'], model['format'])
"""

try:
    import sys
    if sys.version_info.major == 2:
        import urllib as urllib2
        from urllib import quote_plus
    else:
        import urllib2
        from urllib2 import quote_plus
    _use_urllib2 = True
except ImportError:
    import urllib
    import urllib.request
    from urllib.parse import quote_plus
    _use_urllib2 = False

import json


END_POINT = 'https://www.ebi.ac.uk/biomodels/'

def download_from(url):
    """Convenience method reading content from a URL.
    This convenience method uses urlopen on either python 2.7 or 3.x

    Args:
        url (str): the url to read from

    Returns:
        str: the contents of the URL as str
    """
    if _use_urllib2:
        content = urllib2.urlopen(url).read()
    else:
        content = urllib.request.urlopen(url).read().decode('utf-8')
    return content


def download_json(url):
    """Convenience method reading the content of the url as JSON.

    Args:
        url (str): the url to read from

    Returns:
        dict: a python object representing the json content loaded
    """
    content = download_from(url)
    return json.loads(content)


def get_model_info(model_id):
    """Return the model info for the provided `model_id`.

    Args:
        model_id: either an integer, or a valid model id

    Returns:
        a python object describing the model
    """
    if type(model_id) is int:
        model_id = 'BIOMD{0:010d}'.format(model_id)
    result = download_json(END_POINT + model_id + '?format=json')
    return result


def get_files_for_model(model_id):
    """Retrieves the json structure for all files for the given biomodel.
    The structure is of form:

    .. code-block::

        {
            'additional': [
                {'description': 'Auto-generated Scilab file',
                 'fileSize': '3873',
                 'name': 'BIOMD0000000010.sci'},
                 ...
            ],
            'main': [
                {'fileSize': '31568',
                 'name': 'BIOMD0000000010_url.xml'
                }
            ]
        }

    Args:
        model_id: either an integer, or a valid model id

    Returns:
        json structure
    """
    if type(model_id) is int:
        model_id = 'BIOMD{0:010d}'.format(model_id)
    result = download_json(END_POINT + 'model/files/' + model_id + '?format=json')
    return result


def get_content_for_model(model_id, file_name=None):
    """Downloads the specified file from biomodels

    Args:
        model_id: either an integer, or a valid model id
        file_name: the filename to download (or None, to download the main file)

    Returns:
        the content of the specified file
    """
    if type(model_id) is int:
        model_id = 'BIOMD{0:010d}'.format(model_id)
    if file_name is None:
        file_name = get_files_for_model(model_id)['main'][0]['name']
    return download_from(END_POINT + 'model/download/' + model_id + '?filename=' + urllib.parse.quote(file_name))


def search_for_model(query, offset=0, num_results=10, sort='id-asc'):
    """Queries the biomodel database
    Queries the database, for information about the query system see: https://www.ebi.ac.uk/biomodels-static/jummp-biomodels-help/model_search.html

    Example:
        >>> search_for_model('glycolysis')
        [...,
            {
            'format': 'SBML',
            'id': 'BIOMD0000000206',
            'lastModified': '2012-07-04T23:00:00Z',
            'name': 'Wolf2000_Glycolytic_Oscillations',
            'submissionDate': '2008-11-27T00:00:00Z',
            'submitter': 'Harish Dharuri',
            'url': 'https://www.ebi.ac.uk/biomodels/BIOMD0000000206'
            }
        ]
        Note by default, it will include only manually curated models, to obtain Non-curated models you would use:
        >>> search_for_model('Hodgkin AND curationstatus:"Non-curated"')
        [...,
            {
            'format': 'SBML',
            'id': 'MODEL1006230012',
            'lastModified': '2012-02-02T00:00:00Z'',
            'name': 'Stewart2009_ActionPotential_PurkinjeFibreCells',
            'submissionDate': '2010-06-22T23:00:00Z',
            'submitter': 'Camille Laibe',
            'url': 'https://www.ebi.ac.uk/biomodels/MODEL1006230012'
            }
        ]

    Args:
        query: the query to use (it will be encoded with quote_plus before send out, so it is safe to use spaces)
        offset: offset (defaults to 0)
        num_results: number of results to obtain (defaults to 10)
        sort: sort criteria to be used (defaults to id-asc)

    Returns:
        the search result as [{}]
    """
    url = END_POINT + 'search?query=' + quote_plus(query)
    url += '&offset=' + str(offset)
    url += "&numResults=" + str(num_results)
    url += '&sort=' + sort
    result = download_json(url + '&format=json')
    return result['models']