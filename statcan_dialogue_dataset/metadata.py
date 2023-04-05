from datetime import datetime, date
from textwrap import dedent
import warnings

missing_file_explanation = dedent(
    """
    Please download the data at https://mcgill-nlp.github.io/statcan-dialogue-dataset
    and extract it to the data directory.
    """
)


class DataNotFoundWarning(UserWarning):
    pass


class MetadataBase:
    metadata_type = "MetadataBase"

    def __init__(self, code: str, title: str, lang="en"):
        """
        Initialize an object from this class from scratch. This gives more control,
        but it is better to use the from_code or from_title methods if possible,
        as they require fewer arguments.

        Parameters
        ----------
        code : str
            Code that uniquely identifies the object.
        title : str
            Title in the language specified by `lang` describing the object.
        lang : str, default "en"
            The language of the object. Defaults to "en".
        """
        if lang not in ["en", "fr"]:
            raise ValueError("lang must be either 'en' or 'fr'")

        self.code = str(code)
        self.title = title
        self.lang = lang

    def __repr__(self):
        args = ", ".join([f"{k}={repr(v)}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other) and self.__dict__ == other.__dict__

    @classmethod
    def from_code(cls, code: str, lang="en"):
        """
        Creates an instance of this class given a code that uniquely identifies the object.
        For surveys, this is the record number. For tables, this is the product ID (PID).

        Parameters
        ----------
        code : str
            The code of the instance to return.
        lang : str, default "en"
            The language of the instance to return.

        Notes
        -----
        If the code is not found, this will return an object with title=None.
        """
        from ._data.code_sets_map import code_sets_map

        code = str(code)

        title = code_sets_map[lang][cls.metadata_type]["code_to_title"].get(code, None)
        if title is None:
            warnings.warn(
                f"{cls.metadata_type} with code '{code}' not found in language '{lang}'.",
                DataNotFoundWarning,
            )

        return cls(code, title, lang)

    @classmethod
    def from_title(cls, title: str, lang="en"):
        """
        Creates an instance of this class using the given title.

        Parameters
        ----------
        title : str
            The exact title of the instance to return in the specified language.
        lang : str, default "en"
            The language of the instance to return.

        Notes
        -----
        If the title is not found, this will return an object with code -1.
        """
        from ._data.code_sets_map import code_sets_map

        title = str(title)

        code = code_sets_map[lang][cls.metadata_type]["title_to_code"].get(title, -1)
        if code == -1:
            warnings.warn(
                f"{cls.metadata_type} with title '{title}' not found in language '{lang}'.",
                DataNotFoundWarning,
            )

        return cls(code, title, lang)

    def is_valid(self):
        """
        Check if the object is valid. This is the case if the code is not -1 and the title is not None.

        Returns
        -------
        bool
            True if the object is valid, False otherwise.
        """
        return self.code != -1 and self.title != None

    def to_dict(self):
        """
        Returns
        -------
        dict
            A copy of the dictionary representation of the object.
        """
        from copy import deepcopy

        return deepcopy(self.__dict__)

    @classmethod
    def list_all(cls, lang="en", return_as="object", code_length=None):
        """
        List all possible objects belonging to this class. For example, a list of all
        Surveys on Statcan.

        Parameters
        ----------
        lang : str, default "en"
            The language of the instance to return.
        return_as : str, default "object"
            The type of object to return. Must be either "object", "title" or "code".
        code_length : int, default None
            Filter the list to only include objects with a code of this length. If None,
            all objects are returned.

        Returns
        -------
        list
            A list of all possible objects belonging to this class. For example, a list of all
            Surveys on Statcan.
        """
        from ._data.code_sets_map import code_sets_map

        lst = [
            cls.from_code(code, lang=lang)
            for code in code_sets_map[lang][cls.metadata_type]["code_to_title"]
            if code_length is None or len(code) == code_length
        ]

        if return_as == "object":
            return lst
        elif return_as == "title":
            return [o.title for o in lst]
        elif return_as == "code":
            return [o.code for o in lst]
        else:
            raise ValueError(f"return_as must be either 'object', 'title' or 'code'")

    def _get_metadata_from_hierarchy(
        self,
        code_to_obj: "function",
        code_key: str,
        code_type: str,
        return_as: str,
        lang: str,
        warn_if_not_found: bool,
    ):
        from ._data.metadata_hierarchy import metadata_hierarchy

        codes = metadata_hierarchy[code_key].get(self.code, [])

        if warn_if_not_found and len(codes) == 0:
            warnings.warn(
                f"No {code_type} found for {self.metadata_type} '{self.code}'",
                DataNotFoundWarning,
                stacklevel=2,
            )

        lang = lang or self.lang

        if return_as == "code":
            return codes

        objs = [code_to_obj(code, lang) for code in codes]
        if return_as == "object":
            return objs
        elif return_as == "title":
            return [o.title for o in objs]
        else:
            raise ValueError(f"return_as must be either 'object', 'title' or 'code'")


class Frequency(MetadataBase):
    metadata_type = "frequency"


class Survey(MetadataBase):
    """
    StatCan surveys are used to create data tables.
    A list can be found here: https://www.statcan.gc.ca/en/survey/list
    """

    metadata_type = "survey"

    def get_subjects(self, return_as="object", lang=None, warn_if_not_found=True):
        """
        Get the subjects associated with the data tables created from this survey.

        Parameters
        ----------
        return_as : str, default "object"
            How to represent the subjects inside the returned list. Must be either "object",
            "title" or "code".
        lang : str, default None
            The language of the subjects to return. If None, the language of the survey will be
            used.
        warn_if_not_found : bool, default True
            Whether to warn if no subjects are found.

        Returns
        -------
        list of {Subject, str}
            A list of Subject objects, or of the title/code in str. If not found, the list will
            be empty.
        """
        return super()._get_metadata_from_hierarchy(
            code_to_obj=Subject.from_code,
            code_key="survey_to_subject",
            code_type="subject",
            return_as=return_as,
            lang=lang,
            warn_if_not_found=warn_if_not_found,
        )

    def get_tables(self, return_as="object", lang=None, warn_if_not_found=True):
        """
        Get the data tables created from this survey.

        Parameters
        ----------
        return_as : str, default "object"
            How to represent the tables inside the returned list. Must be either "object", "title"
            or "code".
        lang : str, default None
            The language of the table. Must be either "en" or "fr", or None to use the default
            language.
        warn_if_not_found : bool, default True
            Whether to print a warning if no table are found.

        Returns
        -------

        list of {Table, str}
            A list of Table objects, or of the title/code in str. If not found, the list will be
            empty.
        """
        return super()._get_metadata_from_hierarchy(
            code_to_obj=Table.from_code,
            code_key="survey_to_pid",
            code_type="table",
            return_as=return_as,
            lang=lang,
            warn_if_not_found=warn_if_not_found,
        )

    def get_url(self):
        """
        Get the URL on statcan's website containing information about this survey.

        Returns
        -------
        str
            The URL on statcan's website containing information about this survey.
        """
        path = "p2SV" if self.lang == "fr" else "p2SV_f"
        return f"https://www23.statcan.gc.ca/imdb/{path}.pl?Function=getSurvey&SDDS={self.code}"


class Subject(MetadataBase):
    """
    StatCan data tables are grouped in one of many subjects, such as "Energy", "Health", etc.
    A list can be found here: https://www150.statcan.gc.ca/n1/en/subjects
    """

    metadata_type = "subject"

    def get_surveys(self, return_as="object", lang=None, warn_if_not_found=True):
        """
        Get the surveys that belong to this subject, which are parsed from the full metadata.

        Parameters
        ----------
        return_as : str, default "object"
            How to represent the surveys inside the returned list. Must be either "object",
            "title" or "code".
        lang : str, default None
            The language of the survey. Must be either "en" or "fr", or None to use the default
            language.
        warn_if_not_found : bool, default True
            Whether to print a warning if no surveys are found.

        Returns
        -------
        list of {Survey, str}
            A list of Survey objects, or of the title/code in str. If not found, the list will
            be empty.
        """

        return super()._get_metadata_from_hierarchy(
            code_to_obj=Survey.from_code,
            code_key="subject_to_survey",
            code_type="survey",
            return_as=return_as,
            lang=lang,
            warn_if_not_found=warn_if_not_found,
        )

    def get_url(self):
        """
        Get the URL on statcan's website containing information about this subject.

        Returns
        -------
        str
            The URL on statcan's website containing information about this subject.
        """
        path = self.title.lower().replace(" ", "_")
        return f"https://www150.statcan.gc.ca/n1/{self.lang}/subjects/{path}"


class Table:
    def __init__(
        self,
        code: str,
        title: str,
        subjects: "list[Subject]",
        surveys: "list[Survey]",
        frequency: Frequency,
        start_date: date,
        end_date: date,
        release_time: datetime,
        archive_info: str,
        lang="en",
        dimensions: list = None,
        footnotes: list = None,
    ):
        """
        A StatCan data table. A list of tables can be found here:
        https://www150.statcan.gc.ca/n1/en/type/data

        This class is meant to help work with a table's metadata, and not to actually download the data.
        To do that, use the download_full_tables() function. To load the content of a table as a CSV, use
        the sdd.load_table() function.

        You can find useful methods to work with the table's metadata in the MetadataBase class. For
        example, to get the surveys that belong to this table, use the get_surveys() method. To get
        the subjects that belong to this table, use the get_subjects() method.

        Parameters
        ----------
        code : str
            The product ID (PID) of the table. Must be 8 or 10 digits (will be truncated to 8).
        title : str
            The title of the table.
        subjects : list of Subject
            The subjects categorizing this table.
        surveys : list of Survey
            The surveys used to create this table.
        frequency : Frequency
            A Frequency object representing how often the table is updated.
        start_date : datetime.date
            Reference period start date for the table.
        end_date : datetime.date
            Reference period end date for the table.
        release_time : datetime.datetime
            Time the table was released.
        archive_info : str
            Information about whether the table is archived or not.
        lang : str, default 'en'
            The language of the table. Must be either "en" or "fr".
        dimensions : list, optional
            The dimensions of the table, and possibly contains a `member` key holding
            information about the member items.
        footnotes : list, optional
            The footnotes of the table, optional.

        Returns
        -------
        Table
            A table object.

        Notes
        -----
        The dimensions and footnotes are unprocessed JSON. This is the unprocessed JSON but may
            be changed in the future.
        More information about Tables here:
            https://www.statcan.gc.ca/en/developers/wds/user-guide
        """

        if lang not in ["en", "fr"]:
            raise ValueError("lang must be either 'en' or 'fr'")

        if len(code) not in [8, 10]:
            raise ValueError("`code` must be either 8 or 10 characters long")

        self.code = str(code)[:8]
        self.title = title
        self.subjects = subjects
        self.surveys = surveys
        self.frequency = frequency
        self.lang = lang
        self.start_date = start_date
        self.end_date = end_date
        self.release_time = release_time
        self.archive_info = archive_info
        self.dimensions = dimensions
        self.footnotes = footnotes

    @staticmethod
    def __display_obj(obj):
        if isinstance(obj, list) and len(obj) > 0:
            return "[...]"
        elif isinstance(obj, dict) and len(obj) > 0:
            return "{...}"
        else:
            return repr(obj)

    def __repr__(self):
        args = ", ".join(
            [f"{k}={self.__display_obj(v)}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({args})"

    def is_valid(self):
        """
        Checks if this table is valid by if the title is not None and the code is not -1.
        """
        return self.code != -1 and self.title != None

    @classmethod
    def from_code(
        cls, code: str, lang="en", include_members=False, include_footnotes=False
    ):
        """
        Creates a Table object from a table code (PID).

        Parameters
        ----------
        code : str
            The product ID (PID) of the table. Must be 8 or 10 digits (which would be truncated
            to 8).
        lang : str, default 'en'
            The language of the table. Must be either "en" or "fr".
        include_members : bool, default False
            Whether to include the member items of the table inside the `dimensions` attribute.
        include_footnotes : bool, default False
            Whether to include the footnotes of the table.

        Returns
        -------
        Table
            A table object. If the code is not found, the title will be None.
        """
        from ._data.basic_info import basic_info

        code = str(code)

        if code not in basic_info:
            warnings.warn(f"Table with code {code} not found", DataNotFoundWarning)
            return Table(code, None, [], [], None, None, None, None, None, lang=lang)

        subject_codes = basic_info[code]["subject_code"] or []
        survey_codes = basic_info[code]["survey_code"] or []

        subjects = [Subject.from_code(code, lang) for code in subject_codes]
        surveys = [Survey.from_code(code, lang) for code in survey_codes]

        dims = basic_info[code]["dimensions"]
        notes = None

        if include_members:
            from ._data.members import members

            dims = members.get(code, [])

        if include_footnotes:
            from ._data.footnotes import footnotes

            notes = footnotes.get(code, [])

        release_time = datetime.strptime(
            basic_info[code]["release_time"], "%Y-%m-%dT%H:%M"
        )

        return Table(
            code=code,
            title=basic_info[code][f"title_{lang}"],
            subjects=subjects,
            surveys=surveys,
            frequency=Frequency.from_code(basic_info[code]["frequency_code"], lang),
            start_date=date.fromisoformat(basic_info[code]["start_date"]),
            end_date=date.fromisoformat(basic_info[code]["end_date"]),
            release_time=release_time,
            archive_info=basic_info[code][f"archive_{lang}"],
            dimensions=dims,
            footnotes=notes,
            lang=lang,
        )

    @classmethod
    def from_title(
        cls, title: str, lang="en", include_members=False, include_footnotes=False
    ):
        """
        Creates a Table object from a title.

        Parameters
        ----------
        title: str
            The title of the table.
        lang : str, default 'en'
            The language of the table. Must be either "en" or "fr".
        include_members : bool, default False
            Whether to include the member items of the table inside the `dimensions` attribute.
        include_footnotes : bool, default False
            Whether to include the footnotes of the table.

        Returns
        -------
        Table
            A table object. If the title is not found, the code will be -1.
        """
        from ._data.title_to_pid import title_to_pid

        if title not in title_to_pid[lang]:
            return Table(-1, title, [], [], lang=lang)

        pid = title_to_pid[lang][title]

        return cls.from_code(pid, lang, include_members, include_footnotes)

    def get_subjects(self, return_as="object"):
        """
        Get the subjects associated with this table.

        Parameters
        ----------
        return_as : str, optional
            The type of object to return. Must be either "object" or "code".

        Returns
        -------
        list of Subject
            The subjects categorizing this table.
        """
        if return_as == "object":
            return self.subjects
        elif return_as == "code":
            return [subject.code for subject in self.subjects]
        elif return_as == "title":
            return [subject.title for subject in self.subjects]
        else:
            raise ValueError("return_as must be either 'object', 'code', or 'title'")

    def get_surveys(self, return_as="object"):
        """
        Get the surveys associated with this table.

        Parameters
        ----------
        return_as : str, optional
            The type of object to return. Must be either "object" or "code".

        Returns
        -------
        list of Survey
            The surveys used to create this table.
        """
        if return_as == "object":
            return self.surveys
        elif return_as == "code":
            return [survey.code for survey in self.surveys]
        elif return_as == "title":
            return [survey.title for survey in self.surveys]
        else:
            raise ValueError("return_as must be either 'object', 'code', or 'title'")

    @staticmethod
    def get_ignored_codes():
        """
        Get the product IDs (PIDs) that are not included in the dataset.

        Returns
        -------
        set of str
            Product IDs (PIDs) that are not included in the dataset.
        """
        return {
            "12100037",
            "12100147",
            "12100148",
            "12100149",
            "12100150",
            "12100151",
            "12100152",
            "12100153",
            "12100154",
            "12100155",
            "12100156",
            "13100157",
            "13100412",
            "13100575",
            "13100598",
            "13100769",
            "17100062",
            "22100102",
            "32100265",
            "36100293",
        }

    def get_url(self):
        """
        Get the URL on statcan's website containing information about this table.

        Returns
        -------
        str
            The URL on statcan's website containing information about this table.

        Examples
        --------
        >>> Table.from_code("36100293").get_url()
        'https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610029301'

        """
        return f"https://www150.statcan.gc.ca/t1/tbl1/{self.lang}/tv.action?pid={self.code}01"

    def to_dict(self):
        di = self.__dict__.copy()
        for k in di:
            if isinstance(di[k], datetime) or isinstance(di[k], date):
                di[k] = di[k].isoformat()
            if isinstance(di[k], list):
                di[k] = [i.to_dict() if hasattr(i, "to_dict") else i for i in di[k]]

        return di

    @staticmethod
    def key_to_prefix():
        """
        Get a dictionary mapping the keys of the table's `to_dict` method to their prefixes.

        Returns
        -------
        dict of str
            A dictionary mapping the keys of the table's `to_dict` method to their prefixes.
        """
        return {
            "en": {
                "title": "Title",
                "subjects": "Subjects",
                "surveys": "Surveys",
                "frequency": "Update Frequency",
                "start_date": "Start Date",
                "end_date": "End Date",
                "release_time": "Release Time",
                "archive_info": "Archive Status",
                "dimensions": "Dimensions",
                "footnotes": "Footnotes",
            },
            "fr": {
                "title": "Titre",
                "subjects": "Sujets",
                "surveys": "Enquêtes",
                "frequency": "Fréquence de mise à jour",
                "start_date": "Date de début",
                "end_date": "Date de fin",
                "release_time": "Heure de publication",
                "archive_info": "Statut d'archivage",
                "dimensions": "Dimensions",
                "footnotes": "Notes de bas",
            },
        }

    def to_basic_info_text(self, omitted_attrs=("code", "lang", "footnotes")):
        """
        Get a human-readable textual representation of the table's basic information,
        omitting the product ID (PID), member items and footnotes.


        Parameters
        ----------
        omitted_attrs : tuple of str, default ('code', 'lang', 'footnotes')
            Attributes to omit from the output.

        Returns
        -------
        str
            A human-readable textual representation.

        Notes
        -----
        This formats the basic information slightly differently from the paper, as
        the order is different, and some extra information is included.
        """
        k2p = self.key_to_prefix()[self.lang]
        basic_info = []

        for key, value in self.__dict__.items():
            if key in ["start_date", "end_date", "release_time"]:
                value = value.isoformat()
            elif key in ["subjects", "surveys"]:
                value = ", ".join([v.title for v in value])
            elif key == "frequency":
                value = value.title
            elif key == "dimensions":
                value = ", ".join(
                    [v[f"dimensionName{self.lang.capitalize()}"] for v in value]
                )
            elif key in omitted_attrs:
                continue
            basic_info.append(f"{k2p[key]}: {value}")

        return "\n".join(basic_info)
