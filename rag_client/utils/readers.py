"""Document reader implementations for various file formats."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from orgparse import OrgNode


class MailParser(BaseReader):
    """MailParser
    
    Extract text from .eml email files.
    Records only From, Date, Subject headers and plain text body.
    Ignores other headers, HTML parts, and attachments.
    """
    
    def load_data(self, file, extra_info: dict[str, Any] | None = None) -> list[Document]:
        """Parse .eml file and extract essential content for indexing."""
        import email
        from datetime import datetime
        from email import policy
        from email.parser import BytesParser
        
        documents: list[Document] = []
        extra_info = extra_info or {}
        
        # Parse the email file
        with open(file, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract essential headers
        from_header = msg.get('From', '')
        date_header = msg.get('Date', '')
        subject_header = msg.get('Subject', '')
        
        # Build metadata
        doc_extra_info = deepcopy(extra_info)
        doc_extra_info['email_from'] = from_header
        doc_extra_info['email_date'] = date_header
        doc_extra_info['email_subject'] = subject_header
        doc_extra_info['filename'] = str(file)
        
        # Try to parse date into a more usable format
        if date_header:
            try:
                from email.utils import parsedate_to_datetime
                parsed_date = parsedate_to_datetime(date_header)
                doc_extra_info['email_timestamp'] = parsed_date.isoformat()
            except:
                pass  # Keep original date string if parsing fails
        
        # Extract plain text body
        body_parts: list[str] = []
        
        # Handle multipart messages
        if msg.is_multipart():
            for part in msg.walk():
                # Skip non-text parts
                if part.get_content_type() != 'text/plain':
                    continue
                
                # Get the payload, handle encoding
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        # Try to decode with charset from part
                        charset = part.get_content_charset() or 'utf-8'
                        text = payload.decode(charset, errors='replace')
                        body_parts.append(text)
                except Exception as e:
                    # Skip parts that can't be decoded
                    continue
        else:
            # Single part message
            if msg.get_content_type() == 'text/plain':
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or 'utf-8'
                        text = payload.decode(charset, errors='replace')
                        body_parts.append(text)
                except:
                    pass
        
        # Combine all text parts
        body_text = '\n'.join(body_parts).strip()
        
        # Create a formatted text representation
        formatted_text = f"""From: {from_header}
Date: {date_header}
Subject: {subject_header}

{body_text}"""
        
        # Create document with the email content
        documents.append(
            Document(
                text=formatted_text,
                metadata=doc_extra_info,
            )
        )
        
        return documents


class OrgReader(BaseReader):
    """OrgReader
    
    A parser for Org-mode formatted text files.
    
    Extracts text content from .org files and preserves structure
    information in metadata (headings, levels).
    """
    
    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        """Load data from an Org file.
        
        Args:
            file: Path to the Org file.
            extra_info: Optional dictionary of additional metadata.
            
        Returns:
            List of Document objects extracted from the Org file.
        """
        from orgparse import load as org_load
        
        extra_info = extra_info or {}
        documents: list[Document] = []
        
        # Parse the org file
        org_doc = org_load(str(file))
        
        # Extract text from all nodes
        def extract_node_text(node: OrgNode, level: int = 0) -> list[tuple[str, dict[str, Any]]]:
            """Recursively extract text from org nodes.
            
            Returns list of (text, metadata) tuples.
            """
            results: list[tuple[str, dict[str, Any]]] = []
            
            # Get node content
            heading = node.heading if hasattr(node, 'heading') else ""
            body = node.body if hasattr(node, 'body') else ""
            
            # Combine heading and body
            text_parts: list[str] = []
            if heading:
                text_parts.append(heading)
            if body and body.strip():
                text_parts.append(body)
            
            if text_parts:
                text = "\n".join(text_parts)
                
                # Create metadata for this node
                node_metadata = deepcopy(extra_info)
                node_metadata['filename'] = str(file)
                node_metadata['org_level'] = level
                if heading:
                    node_metadata['org_heading'] = heading
                    
                # Add tags if present
                if hasattr(node, 'tags') and node.tags:
                    node_metadata['org_tags'] = list(node.tags)
                    
                # Add TODO state if present
                if hasattr(node, 'todo') and node.todo:
                    node_metadata['org_todo'] = node.todo
                    
                results.append((text, node_metadata))
            
            # Process children
            if hasattr(node, 'children'):
                for child in node.children:
                    results.extend(extract_node_text(child, level + 1))
                    
            return results
        
        # Extract all text nodes
        text_nodes = extract_node_text(org_doc)
        
        # Create documents from text nodes
        for text, metadata in text_nodes:
            if text.strip():  # Only create document if there's actual content
                documents.append(
                    Document(
                        text=text,
                        metadata=metadata,
                    )
                )
        
        # If no nodes were extracted, create a single document with all text
        if not documents:
            full_text = org_doc.get_body() if hasattr(org_doc, 'get_body') else str(org_doc)
            if full_text and full_text.strip():
                fallback_metadata = deepcopy(extra_info)
                fallback_metadata['filename'] = str(file)
                documents.append(
                    Document(
                        text=full_text,
                        metadata=fallback_metadata,
                    )
                )
        
        return documents