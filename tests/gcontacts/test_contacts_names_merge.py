"""
Lesson #301 (Market Pulse R88/R90): a `manage_contact` update that gave only one part of
the name replaced the whole `names` block, so the other part was erased. The fix is a
read-merge-write helper, `_merge_names`, used by the single update and the batch update.

Covers:
- the helper alone: keep the missing part, overlay the given one, clear on "", keep the
  other writable name fields, drop output-only fields, prefer the primary entry, no names yet
- the single `manage_contact` update sends the merged block to `updateContact`
- the batch `manage_contacts_batch` update fetches names too and sends the merged block
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gcontacts import contacts_tools  # noqa: E402
from gcontacts.contacts_helpers import _merge_names  # noqa: E402


EXISTING = [
    {
        "metadata": {"primary": True, "source": {"type": "CONTACT"}},
        "displayName": "Zzq Testperson",
        "givenName": "Zzq",
        "familyName": "Testperson",
        "middleName": "Q",
        "honorificPrefix": "Dr.",
    }
]


class TestMergeNames:
    def test_family_only_keeps_given(self):
        out = _merge_names(EXISTING, None, "Testperson2")
        assert out == [
            {
                "givenName": "Zzq",
                "familyName": "Testperson2",
                "middleName": "Q",
                "honorificPrefix": "Dr.",
            }
        ]

    def test_given_only_keeps_family(self):
        out = _merge_names(EXISTING, "Zzq2", None)
        assert out[0]["givenName"] == "Zzq2"
        assert out[0]["familyName"] == "Testperson"

    def test_both_given_overlays_both(self):
        out = _merge_names(EXISTING, "A", "B")
        assert out[0]["givenName"] == "A"
        assert out[0]["familyName"] == "B"
        assert out[0]["middleName"] == "Q"

    def test_empty_string_clears_that_part(self):
        out = _merge_names(EXISTING, "", None)
        assert out[0]["givenName"] == ""
        assert out[0]["familyName"] == "Testperson"

    def test_output_only_fields_are_dropped(self):
        out = _merge_names(EXISTING, None, "X")
        assert "displayName" not in out[0]
        assert "metadata" not in out[0]

    def test_primary_entry_is_preferred(self):
        existing = [
            {"metadata": {"primary": False}, "givenName": "Other", "familyName": "Entry"},
            {"metadata": {"primary": True}, "givenName": "Zzq", "familyName": "Testperson"},
        ]
        out = _merge_names(existing, None, "New")
        assert out == [{"givenName": "Zzq", "familyName": "New"}]

    def test_first_entry_when_none_is_primary(self):
        existing = [{"givenName": "Zzq", "familyName": "Testperson"}, {"givenName": "B"}]
        out = _merge_names(existing, "A", None)
        assert out == [{"givenName": "A", "familyName": "Testperson"}]

    def test_contact_without_names(self):
        assert _merge_names([], None, "Only") == [{"familyName": "Only"}]
        assert _merge_names(None, "Only", None) == [{"givenName": "Only"}]

    def test_always_one_entry(self):
        assert len(_merge_names(EXISTING, "A", "B")) == 1

    def test_stale_unstructured_name_is_not_sent_back(self):
        # mp-reviewer R90 #1: the old request never sent unstructuredName; a stale one next to
        # a changed familyName could win at Google and the tool would still say "updated".
        existing = [
            {
                "metadata": {"primary": True, "source": {"type": "CONTACT"}},
                "unstructuredName": "Zzq Testperson",
                "givenName": "Zzq",
                "familyName": "Testperson",
            }
        ]
        out = _merge_names(existing, None, "Testperson2")
        assert out == [{"givenName": "Zzq", "familyName": "Testperson2"}]

    def test_profile_name_is_never_copied_into_the_contact(self):
        # Astra a-20260926-3mvh #2 / mp-reviewer R90 #2: a contact saved with only an address,
        # linked to a Google account, carries the PROFILE's name as primary. A family-only update
        # must not write the profile's given name (or anything else) into the contact.
        existing = [
            {
                "metadata": {"primary": True, "source": {"type": "PROFILE"}},
                "givenName": "Profile",
                "familyName": "Person",
                "middleName": "P",
            }
        ]
        assert _merge_names(existing, None, "Testperson2") == [{"familyName": "Testperson2"}]

    def test_contact_source_entry_wins_over_a_primary_profile_entry(self):
        existing = [
            {
                "metadata": {"primary": True, "source": {"type": "PROFILE"}},
                "givenName": "Profile",
                "familyName": "Person",
            },
            {
                "metadata": {"primary": False, "source": {"type": "CONTACT"}},
                "givenName": "Zzq",
                "familyName": "Testperson",
            },
        ]
        assert _merge_names(existing, None, "Testperson2") == [
            {"givenName": "Zzq", "familyName": "Testperson2"}
        ]

    def test_entries_without_source_count_as_the_contacts_own(self):
        existing = [{"givenName": "Zzq", "familyName": "Testperson"}]
        assert _merge_names(existing, "A", None) == [{"givenName": "A", "familyName": "Testperson"}]

    def test_explicit_contact_entry_beats_a_primary_sourceless_one(self):
        # Astra a-20260926-45q6: missing metadata does not establish ownership
        existing = [
            {"metadata": {"primary": True}, "givenName": "Nosource", "familyName": "Entry"},
            {"metadata": {"source": {"type": "CONTACT"}}, "givenName": "Zzq", "familyName": "Testperson"},
        ]
        assert _merge_names(existing, None, "X") == [{"givenName": "Zzq", "familyName": "X"}]


class TestBuilderEmitsNamesForEmptyString:
    # Astra a-20260926-3mvh #1: "" is a value (it clears that part); the builder used to emit
    # `names` only for a truthy value, so a clear-one-part update never reached the merge.
    def test_empty_given_alone_emits_names(self):
        from gcontacts.contacts_tools import _build_person_body

        body = _build_person_body(given_name="", family_name=None)
        assert body["names"] == [{"givenName": "", "familyName": ""}]

    def test_no_name_given_emits_no_names(self):
        from gcontacts.contacts_tools import _build_person_body

        assert "names" not in _build_person_body(given_name=None, family_name=None, notes="x")


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _people_service(get_result=None, batch_get_result=None):
    """A People API stub: get(), getBatchGet(), updateContact(), batchUpdateContacts()."""
    service = MagicMock()
    people = service.people.return_value
    people.get.return_value.execute.return_value = get_result or {}
    people.getBatchGet.return_value.execute.return_value = batch_get_result or {}
    people.updateContact.return_value.execute.return_value = {
        "resourceName": "people/c1",
        "names": [{"givenName": "Zzq", "familyName": "Testperson2"}],
    }
    people.batchUpdateContacts.return_value.execute.return_value = {"updateResult": {}}
    return service


def _unwrap(tool):
    # The tools are wrapped by the server decorators; the tests call the plain coroutine
    # exactly as the other contacts tests do (the wrapper keeps the original on `fn`/`__wrapped__`).
    for attr in ("fn", "__wrapped__"):
        inner = getattr(tool, attr, None)
        if inner is not None:
            return _unwrap(inner)
    return tool


class TestSingleUpdateMerges:
    def test_family_only_update_sends_merged_names(self):
        service = _people_service(
            get_result={"resourceName": "people/c1", "etag": "e1", "names": EXISTING}
        )
        fn = _unwrap(contacts_tools.manage_contact)
        _run(
            fn(
                service,
                "zzq@example.invalid",
                action="update",
                contact_id="people/c1",
                family_name="Testperson2",
            )
        )
        kwargs = service.people.return_value.updateContact.call_args.kwargs
        assert kwargs["updatePersonFields"] == "names"
        assert kwargs["body"]["names"] == [
            {
                "givenName": "Zzq",
                "familyName": "Testperson2",
                "middleName": "Q",
                "honorificPrefix": "Dr.",
            }
        ]
        assert kwargs["body"]["etag"] == "e1"

    def test_empty_given_alone_clears_only_the_given_name(self):
        service = _people_service(
            get_result={"resourceName": "people/c1", "etag": "e1", "names": EXISTING}
        )
        fn = _unwrap(contacts_tools.manage_contact)
        _run(fn(service, "zzq@example.invalid", action="update", contact_id="people/c1", given_name=""))
        kwargs = service.people.return_value.updateContact.call_args.kwargs
        assert kwargs["updatePersonFields"] == "names"
        assert kwargs["body"]["names"] == [
            {"givenName": "", "familyName": "Testperson", "middleName": "Q", "honorificPrefix": "Dr."}
        ]

    def test_412_retry_merges_from_the_refetched_contact(self):
        # mp-reviewer R90 #4: on an etag conflict the contact is fetched again and the merge
        # must use that fresh copy, not the first one.
        from googleapiclient.errors import HttpError

        first = {"resourceName": "people/c1", "etag": "e1", "names": EXISTING}
        second = {
            "resourceName": "people/c1",
            "etag": "e2",
            "names": [
                {
                    "metadata": {"primary": True, "source": {"type": "CONTACT"}},
                    "givenName": "Zzq-renamed",
                    "familyName": "Testperson",
                }
            ],
        }
        service = _people_service()
        people = service.people.return_value
        people.get.return_value.execute.side_effect = [first, second]
        resp = MagicMock()
        resp.status = 412
        resp.reason = "Precondition Failed"
        people.updateContact.return_value.execute.side_effect = [
            HttpError(resp, b"conflict"),
            {"resourceName": "people/c1", "names": [{"givenName": "Zzq-renamed", "familyName": "Testperson2"}]},
        ]
        fn = _unwrap(contacts_tools.manage_contact)
        _run(
            fn(service, "zzq@example.invalid", action="update", contact_id="people/c1", family_name="Testperson2")
        )
        assert people.updateContact.call_count == 2
        retried = people.updateContact.call_args_list[1].kwargs["body"]
        assert retried["etag"] == "e2"
        assert retried["names"] == [{"givenName": "Zzq-renamed", "familyName": "Testperson2"}]


class TestBatchUpdateMerges:
    def test_batch_names_update_fetches_and_merges(self):
        service = _people_service(
            batch_get_result={
                "responses": [
                    {"person": {"resourceName": "people/c1", "etag": "e1", "names": EXISTING}}
                ]
            }
        )
        fn = _unwrap(contacts_tools.manage_contacts_batch)
        _run(
            fn(
                service,
                "zzq@example.invalid",
                action="update",
                field="names",
                updates=[{"contact_id": "people/c1", "family_name": "Testperson2"}],
            )
        )
        people = service.people.return_value
        assert people.getBatchGet.call_args.kwargs["personFields"] == "metadata,names"
        sent = people.batchUpdateContacts.call_args.kwargs["body"]
        assert sent["contacts"]["people/c1"]["names"] == [
            {
                "givenName": "Zzq",
                "familyName": "Testperson2",
                "middleName": "Q",
                "honorificPrefix": "Dr.",
            }
        ]

    def test_batch_other_field_still_fetches_metadata_only(self):
        service = _people_service(
            batch_get_result={
                "responses": [{"person": {"resourceName": "people/c1", "etag": "e1"}}]
            }
        )
        fn = _unwrap(contacts_tools.manage_contacts_batch)
        _run(
            fn(
                service,
                "zzq@example.invalid",
                action="update",
                field="nicknames",
                updates=[{"contact_id": "people/c1", "nicknames": [{"value": "Z"}]}],
            )
        )
        people = service.people.return_value
        assert people.getBatchGet.call_args.kwargs["personFields"] == "metadata"
